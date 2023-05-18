import numpy as np
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import RandomSampler
import torch.distributed as dist
import random
from common.engine_pertrain import *
from common.datautils import GF5Dataset

from models.HsiMAE import hsimae_15p_204c_stiny_model
import torch


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.samples) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


class SchedulerDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """

    def __init__(self, dataset, shuffle, num_replicas=None, rank=None, seed=0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.samples) for cur_dataset in dataset.datasets])
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        assert self.number_of_datasets == self.num_replicas, 'Invalid dataset numbers.'

    def __len__(self):
        return self.largest_dataset_size

    def __iter__(self):
        cur_dataset = self.dataset.datasets[self.rank]
        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            random.seed(self.seed + self.epoch)
            diff = self.largest_dataset_size - len(cur_dataset)
            padding = (torch.tensor(random.sample(range(len(cur_dataset)), diff)) + push_index_val[self.rank]).tolist()
            indices = (torch.randperm(len(cur_dataset), generator=g) + push_index_val[self.rank]).tolist()
            indices += padding
        else:
            diff = self.largest_dataset_size - len(cur_dataset)
            padding = list(range(len(cur_dataset)))[0:diff]
            indices = list(range(len(cur_dataset))) + padding + push_index_val[self.rank]

        assert len(indices) == self.largest_dataset_size
        return iter(indices)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class BatchSchedulerDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """

    def __init__(self, dataset, batch_size, shuffle, num_replicas=None, rank=None, drop_last=False, seed=0):
        assert batch_size % num_replicas == 0
        self.dataset = dataset
        self.batch_size = batch_size // num_replicas
        self.shuffle = shuffle
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.samples) for cur_dataset in dataset.datasets])
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.seed = seed
        if not self.drop_last:
            self.num_batches = math.ceil(
                math.ceil(self.largest_dataset_size / self.batch_size) / self.num_replicas)  # 计算最大数据集上的分布式batch数
            self.largest_dataset_size = self.num_batches * self.batch_size * self.num_replicas  # 更新补全后的数据集样本数
            self.num_samples = self.largest_dataset_size * self.number_of_datasets  # 计算补全后总样本数
        else:
            self.num_samples = None
            self.num_batches = None
            print('not implemented')

    def __len__(self):
        return self.num_samples // self.num_replicas  # samples nums on per node

    def __iter__(self):
        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        samples = []
        final_samples_list = []
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            for dataset_idx in range(self.number_of_datasets):
                cur_dataset = self.dataset.datasets[dataset_idx]
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                np.random.seed(self.seed + self.epoch * 2)
                diff = self.largest_dataset_size - len(cur_dataset)  # oversample small dataset to largest_dataset_size
                padding = (np.random.choice(range(len(cur_dataset)), diff)+push_index_val[dataset_idx]).tolist()
                indices = (torch.randperm(len(cur_dataset), generator=g) + push_index_val[dataset_idx]).tolist()
                indices += padding

                # diff = self.num_samples - self.largest_dataset_size  # padding to fit calculated sample number
                # padding = (np.random.choice(range(len(cur_dataset)), diff)+push_index_val[dataset_idx]).tolist()
                # indices += padding
                samples.append(indices)
            samples = np.asarray(samples)
        else:
            print('sequential sample is not implemented')
        for b in range(self.num_batches):
            temp = samples[:, b*self.batch_size*self.num_replicas:(b+1)*self.batch_size*self.num_replicas][:, self.rank*self.batch_size:(self.rank+1)*self.batch_size]
            temp = temp.reshape(-1).tolist()  # alternative splicing
            final_samples_list += temp
        return iter(final_samples_list)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class UnifLabelSampler(torch.utils.data.sampler.Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


class DistConcatDataset(ConcatDataset):
    def __len__(self):
        return len(self.cumulative_sizes)*self.cumulative_sizes[0]


class MyFirstDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat((-torch.ones(100000), torch.ones(100000)))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]


class MySecondDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat((torch.ones(150000) * 2, torch.ones(150000) * -2))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]


class MyThirdDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat((torch.ones(200000) * 3, torch.ones(200000) * -3))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]


if __name__ == '__main__':

    # first_dataset = MyFirstDataset()
    # second_dataset = MySecondDataset()
    # third_dataset = MyThirdDataset()
    # concat_dataset = ConcatDataset([first_dataset, second_dataset, third_dataset])

    batch_size = 180
    word_size = 3
    local_rank = 0

    # sample = BatchSchedulerSampler(dataset=concat_dataset,
    #                                batch_size=batch_size),
    # final_samples_list = sample.__iter__()
    # dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
    #                                          sampler=BatchSchedulerSampler(dataset=concat_dataset,
    #                                                                        batch_size=batch_size),
    #                                          batch_size=batch_size,
    #                                          shuffle=False)

    # sampler = SchedulerDistributedSampler(dataset=concat_dataset, shuffle=True, num_replicas=2, rank=1)
    # final_samples_list = sampler.__iter__()
    # dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
    #                                          sampler=sampler,
    #                                          batch_size=batch_size,
    #                                          shuffle=False)
    # epoch = 3
    # for epoch in range(epoch):
    #     sampler.set_epoch(epoch)
    #     for inputs in dataloader:
    #         print(inputs)
    # start = time.time()
    # sampler1 = list(torch.utils.data.sampler.SubsetRandomSampler(
    #     np.random.choice(range(200000), 200)))
    # end = time.time()
    # print(end-start)

    tr_root = r'../data/GF5_patches/train'
    tr_list = os.listdir(tr_root)
    tr_dataset_list = [GF5Dataset(img_root=os.path.join(tr_root, path)) for path in tr_list]
    te_root = r'../data/GF5_patches/test'
    te_list = os.listdir(te_root)
    te_dataset_list = [GF5Dataset(img_root=os.path.join(te_root, path)) for path in te_list]
    tr_concat_dataset = DistConcatDataset(tr_dataset_list)
    te_concat_dataset = DistConcatDataset(te_dataset_list)

    sampler = BatchSchedulerDistributedSampler(dataset=tr_concat_dataset, batch_size=batch_size, shuffle=True, num_replicas=word_size, rank=local_rank)
    # final_samples_list = list(sampler.__iter__())
    dataloader = torch.utils.data.DataLoader(dataset=tr_concat_dataset,
                                             sampler=sampler,
                                             batch_size=batch_size//word_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=8)
    # plt.figure(0, (40, 10))
    # plt.plot(final_samples_list)
    # plt.show()
    model = hsimae_15p_204c_stiny_model
    model.cuda()
    epoch = 1
    for epoch in range(epoch):
        # start = time.time()
        sampler.set_epoch(epoch)
        for inputs in dataloader:
            # print(inputs.shape)
            pass
        # end = time.time()
        # print(end-start)
