import torch
import os
import sys
import time
import shutil
from tqdm import tqdm
from common.dist_utils import is_main_process, reduce_value
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import math


def simple_trainer(model, dataloader, optimizer, epoch, epoches):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.cuda()
    train_loss = 0
    loss_ = 0
    with tqdm(enumerate(dataloader), total=len(dataloader)) as train_bar:
        for batch, X in enumerate(train_bar):
            # Compute prediction and loss
            model.train()
            X = X[1].to(device)
            _, loss, _ = model(X, 0.5)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            _, loss, _ = model(X, 0.5)
            train_loss += loss.item()
            loss_ = train_loss / (batch + 1)
            train_bar.set_description(f'Epoch[{epoch + 1}/{epoches}]')
            train_bar.set_postfix_str(f'average_loss={loss_}')  # trian_loss={loss.item()},
    return loss_


def test_loop(model, dataloader, epoch, epoches):
    test_loss = 0
    loss_ = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.cuda()
    with torch.no_grad():
        with tqdm(enumerate(dataloader), total=len(dataloader), colour='blue') as test_bar:
            for batch, X in enumerate(test_bar):
                X = X[1].to(device)
                pred, loss, _ = model(X, 0.5)
                test_loss += loss.item()
                test_bar.set_description(f'Valid[{epoch + 1}/{epoches}]')
                loss_ = test_loss / (batch + 1)
                test_bar.set_postfix_str(f'average_loss={loss_},current_loss={loss.item()}')
    return loss_


def train_one_epoch(model, optimizer, dataloader, device, epoch, epoches):
    model.cuda()
    mean_loss = 0
    mean_loss_r = 0
    mean_loss_c = 0
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        # world_size = int(os.environ['WORLD_SIZE'])
        # total = math.ceil(math.ceil(len(dataloader.dataset)/dataloader.batch_size)/world_size) + 1
        # print(len(dataloader.dataset))
        with tqdm(enumerate(dataloader), total=len(dataloader)) as train_bar:
            for batch, X in enumerate(train_bar):
                # Compute prediction and loss
                model.train()
                X = X[1].to(device)
                _, _, loss, loss_r, loss_c = model(X, 0.5)

                # Backpropagation
                loss.backward()
                loss = reduce_value(loss, average=True)
                optimizer.step()
                optimizer.zero_grad()

                mean_loss = (mean_loss * batch + loss.item()) / (batch + 1)
                loss_r = reduce_value(loss_r, average=True)
                mean_loss_r = (mean_loss_r * batch + loss_r.item()) / (batch + 1)
                loss_c = reduce_value(loss_c, average=True)
                mean_loss_c = (mean_loss_c * batch + loss_c.item()) / (batch + 1)

                train_bar.set_description(f'Epoch[{epoch + 1}/{epoches}]')
                train_bar.set_postfix_str(f'avg_loss:{round(mean_loss,3)},r_loss:{round(mean_loss_r,3)},c_loss:{round(mean_loss_c,3)}')

                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss)
                    sys.exit(1)
    else:
        for batch, X in enumerate(dataloader):
            # Compute prediction and loss
            model.train()
            X = X.to(device)
            _, _, loss, loss_r, loss_c = model(X, 0.5)
            # Backpropagation
            loss.backward()
            loss = reduce_value(loss, average=True)
            optimizer.step()
            optimizer.zero_grad()

            mean_loss = (mean_loss * batch + loss.item()) / (batch + 1)
            loss_r = reduce_value(loss_r, average=True)
            mean_loss_r = (mean_loss_r * batch + loss_r.item()) / (batch + 1)
            loss_c = reduce_value(loss_c, average=True)
            mean_loss_c = (mean_loss_c * batch + loss_c.item()) / (batch + 1)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)
    return mean_loss, mean_loss_r, mean_loss_c


@torch.no_grad()
def test_one_epoch(model, dataloader, device, epoch, epoches):
    model.cuda()
    mean_loss = 0
    mean_loss_r = 0
    mean_loss_c = 0
    if is_main_process():
        with tqdm(enumerate(dataloader), total=len(dataloader), colour='#ccffff') as test_bar:
            for batch, X in enumerate(test_bar):
                X = X[1].to(device)
                pred, _, loss, loss_r, loss_c = model(X, 0.5)
                loss = reduce_value(loss, average=True)
                mean_loss = (mean_loss * batch + loss.item()) / (batch + 1)
                loss_r = reduce_value(loss_r, average=True)
                mean_loss_r = (mean_loss_r * batch + loss_r.item()) / (batch + 1)
                loss_c = reduce_value(loss_c, average=True)
                mean_loss_c = (mean_loss_c * batch + loss_c.item()) / (batch + 1)

                test_bar.set_description(f'Valid[{epoch + 1}/{epoches}]')
                test_bar.set_postfix_str(f'avg_loss:{round(mean_loss,3)},r_loss:{round(mean_loss_r,3)},c_loss:{round(mean_loss_c,3)}')

    else:
        for batch, X in enumerate(dataloader):
            X = X.to(device)
            pred, _, loss, loss_r, loss_c = model(X, 0.5)
            loss = reduce_value(loss, average=True)
            mean_loss = (mean_loss * batch + loss.item()) / (batch + 1)
            loss_r = reduce_value(loss_r, average=True)
            mean_loss_r = (mean_loss_r * batch + loss_r.item()) / (batch + 1)
            loss_c = reduce_value(loss_c, average=True)
            mean_loss_c = (mean_loss_c * batch + loss_c.item()) / (batch + 1)

            # s = ssim(X, pred)
            # s = reduce_value(s, average=True)
            # mean_ssim = (mean_ssim * batch + s.item()) / (batch + 1)
            # p = psnr(X, pred)
            # p = reduce_value(p, average=True)
            # mean_psnr = (mean_psnr * batch + p.item()) / (batch + 1)
        #     s = ssim(X.to('cpu'), pred.to('cpu'))
        #     s = reduce_value(s, average=True)
        #     mean_ssim = (mean_ssim * batch + s.item()) / (batch + 1)
        #     p = psnr(X.to('cpu'), pred.to('cpu'))
        #     p = reduce_value(p, average=True)
        #     mean_psnr = (mean_psnr * batch + p.item()) / (batch + 1)
        # del ssim, psnr

    return mean_loss, mean_loss_r, mean_loss_c


def mv(old_path, new_path):
    print(old_path)
    print(new_path)
    filelist = os.listdir(old_path)
    print(filelist)
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        print('src:', src)
        print('dst:', dst)
        shutil.move(src, dst)


def recoder(Rootpath, model, ModelArchive, train_loss, valid_loss):
    assert len(train_loss) == len(valid_loss), 'The length of loss lists are not equal, please check it.'
    if not os.path.exists(Rootpath):
        os.makedirs(Rootpath)
    p = f'{model.name}' + '_' + time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    p = os.path.join(Rootpath, p)
    os.makedirs(p, exist_ok=True)
    with open(p + r'\loss_record.txt', 'w') as f:
        for i in range(len(train_loss)):
            f.write(f'Epoch:{i}, loss:{train_loss[i]}, valid_loss:{valid_loss[i] + 0.2} \n')
    with open(p + r'\model_paremeters.txt', 'w') as f:
        f.write(f'{model.parameters}')
    mv(ModelArchive, p)


if __name__ == '__main__':
    loss = [1, 2, 3, 4, 5, 6, 7, 8]
    loss_ = [1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(10):
        loss.append(i)
        print(loss[i] * 10)
    # epoch = len(loss)
    # recoder(RecordRootPath, hsimae_15p_204c_tiny_model, loss, loss_)
