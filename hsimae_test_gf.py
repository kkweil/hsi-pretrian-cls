import torch
from torch.utils.data import DataLoader
from common.datautils import GF5Dataset
from models.HsiMAE import hsimae_15p_204c_sstiny_model
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

if __name__ == '__main__':
    size = 31
    imp = r'../data/GF5_patches/test/patch' + str(size)
    dataset = GF5Dataset(img_root=imp)
    # data = next(iter(dataset))
    test_dataloader = DataLoader(dataset=dataset,
                                 batch_size=256,
                                 shuffle=True,
                                 num_workers=0)

    # from common.datautils import HSIloader, HSIDataset
    # imp = r'../data/Salinas_corrected.mat'
    # gtp = r'../data/Salinas_gt.mat'
    # imp = r'../data/Indian_pines_corrected.mat'
    # gtp = r'../data/Indian_pines_gt.mat'
    # dataset = HSIloader(img_path=imp, gt_path=gtp, patch_size=size, sample_mode='ratio', train_ratio=0.1,
    #                     sample_points=30, merge=None, rmbg=False)
    # dataset(spectral=False)

    # train_dataset = HSIDataset(dataset.x_train_patch, dataset.gt, dataset.coordinate_train)
    # train_dataloader = DataLoader(test_dataloader, batch_size=256, shuffle=True, pin_memory=True)
    data = next(iter(test_dataloader))
    model = hsimae_15p_204c_sstiny_model
    model.load_state_dict(torch.load(r'runs/exp1/best.pth'))
    # model.load_state_dict(torch.load(r'ModelArchive\mae_pertrain\best.pth'))
    model.eval()

    with torch.no_grad():
        # x, _, _ = model.encoder_forward(data, 0.5)

        out_data, mask, loss, loss_r, loss_c = model(data, 0.5)
        print('loss_r:', loss_r)
        print('loss_center:', loss_c)

    mask = mask.reshape(-1, size, size)

    figure = plt.figure(figsize=(16, 6))
    cols, rows = 10, 3
    band = 1

    for i in range(1, cols + 1):
        figure.add_subplot(rows, cols, i)
        t = data[i].numpy()[band]
        t = (t - t.min()) / (t.max() - t.min())
        plt.imshow(t, cmap='gnuplot2')
        plt.title('orignal')
        plt.axis('off')

    for i in range(1, cols + 1):
        figure.add_subplot(rows, cols, i + cols)
        t = data[i].numpy()[band]
        t = (t - t.min()) / (t.max() - t.min())
        plt.imshow(t * (1 - mask[i].numpy()), cmap='gnuplot2')
        plt.title('mask')
        plt.axis('off')

    for i in range(1, cols + 1):
        figure.add_subplot(rows, cols, i + 2 * cols)
        t = out_data[i].numpy()[band]
        t = (t - t.min()) / (t.max() - t.min())
        plt.imshow(t, cmap='gnuplot2')
        plt.title('re')
        plt.axis('off')
    plt.show()

    # plt.imshow(data.squeeze().numpy()[19])
    # plt.show()
    # plt.imshow(out_data.squeeze().numpy()[19])
    # plt.show()
    ssim = StructuralSimilarityIndexMeasure()
    psnr = PeakSignalNoiseRatio()
    # t_s, t_p = 0, 0
    # data = torch.randn((5, 10, 15, 15))
    # out_data = torch.randn((5, 10, 15, 15))
    # for i in range(len(data)):
    #     t_s = (t_s * i + ssim(data[i].unsqueeze(0), out_data[i].unsqueeze(0))) / (i + 1)
    #     t_p = (t_p * i + psnr(data[i].unsqueeze(0), out_data[i].unsqueeze(0))) / (i + 1)
    print('ssim:', ssim(data, out_data))
    print('psnr:', psnr(data, out_data))

    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 3, 2
    # for i in range(1, cols * rows + 1):
    #     figure.add_subplot(rows, cols, i)
    #     plt.imshow(data[i].numpy()[0])
    # plt.show()
    # figure2 = plt.figure(figsize=(8, 8))
    # cols, rows = 3, 2
    # for i in range(1, cols * rows + 1):
    #     figure2.add_subplot(rows, cols, i)
    #     plt.imshow(out_data[i].numpy()[0])
    # plt.show()

    a = 0
