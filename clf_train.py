from torch import nn
from models.clf import Classfier
from common.datautils import *
from torch.utils.tensorboard import SummaryWriter


def train_epoch(model, dataloader, optimzier, loss_fn):
    if torch.cuda.is_available():
        model.cuda()
    size = len(dataloader.dataset)
    correct = 0
    loss_ = []
    for batch, (X, y) in enumerate(dataloader):
        model.train()
        preds = model(X.cuda())
        loss = loss_fn(preds, y.cuda())
        loss_.append(loss.item())
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        model.eval()
        correct = correct + (model(X.cuda()).argmax(1) == y.cuda()).type(torch.float).sum().item()

        # print(f"loss: {loss.item():>7f}  [{batch * len(X):>5d}/{size:>5d}]")
    accuracy = (correct / size) * 100.0
    print(f"Avg Accuracy：{round(accuracy, 3)}%")
    torch.cuda.empty_cache()
    return loss_, accuracy


def test_epoch(model, dataloader, loss_fn):
    if torch.cuda.is_available():
        model.cuda()
    size = len(dataloader.dataset)
    correct = 0
    loss_ = []
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            preds = model(X.cuda())
            loss = loss_fn(preds, y.cuda())
            loss_.append(loss.item())
            correct = correct + (model(X.cuda()).argmax(1) == y.cuda()).type(torch.float).sum().item()
            # if batch % 100 == 0:
            #     print(f"loss: {loss.item():>7f}  [{batch * len(X):>5d}/{size:>5d}]")
    accuracy = (correct / size) * 100.0
    print(f"Avg Accuracy：{round(accuracy, 3)}%")
    torch.cuda.empty_cache()
    return loss_, accuracy


if __name__ == '__main__':
    # inputs = torch.randn((20, 204, 15, 15))
    class_num = 9
    learning_rate = 1e-3
    epochs = 300
    batch_size = 32
    pretrain = True
    # load hsi data
    # imp = r'../data/Salinas_corrected.mat'
    # gtp = r'../data/Salinas_gt.mat'
    # imp = r'../data/Indian_pines_corrected.mat'
    # gtp = r'../data/Indian_pines_gt.mat'
    imp = r'../data/PaviaU.mat'
    gtp = r'../data/PaviaU_gt.mat'

    dataset = HSIloader(img_path=imp, gt_path=gtp, patch_size=31, sample_mode='ratio', train_ratio=0.1,
                        sample_points=30, merge=None, rmbg=False)
    dataset(spectral=False)

    train_dataset = HSIDataset(dataset.x_train_patch, dataset.gt, dataset.coordinate_train)
    test_dataset = HSIDataset(np.concatenate([dataset.x_test_patch, dataset.x_train_patch]), dataset.gt,
                              np.concatenate([dataset.coordinate_test, dataset.coordinate_train]))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True, pin_memory=True)
    # next(iter(train_dataloader))

    hsiclf_15p_204c_sstiny_model = Classfier(name='hsimae_15p_204c_stiny_model', in_chans=103, class_num=class_num,
                                             encoder_embed_dim=256, encoder_depth=4, encoder_num_heads=8,
                                             mlp_ratio=4., norm_layer=nn.LayerNorm)

    # load masked autoencoder pretrain
    if pretrain:
        pretrain_dict = torch.load(r'runs/exp1/best.pth')
        clf_dict = hsiclf_15p_204c_sstiny_model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items()
                         if (k in clf_dict) and (k not in ['patch_embed.conv2d_1.weight'])}
        # hsiclf_15p_204c_stiny_model.load_state_dict(torch.load(r'../runs/exp1/best.pth'), strict=False)
        clf_dict.update(pretrain_dict)
        hsiclf_15p_204c_sstiny_model.load_state_dict(clf_dict)
    # check device. move model to GPU
    if torch.cuda.is_available():
        hsiclf_15p_204c_sstiny_model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    if pretrain:
        optimizer = torch.optim.AdamW([
            {'params': hsiclf_15p_204c_sstiny_model.patch_embed.parameters()},
            {'params': hsiclf_15p_204c_sstiny_model.pos_embed.parameters(), 'lr': 8e-5},
            {'params': hsiclf_15p_204c_sstiny_model.blocks.parameters(), 'lr': 8e-5},
            {'params': hsiclf_15p_204c_sstiny_model.norm.parameters(), 'lr': 8e-5},
            {'params': hsiclf_15p_204c_sstiny_model.fc.parameters()},
            {'params': hsiclf_15p_204c_sstiny_model.linear_comb.parameters()},
            {'params': hsiclf_15p_204c_sstiny_model.head.parameters()},

        ], lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(hsiclf_15p_204c_sstiny_model.parameters(), lr=learning_rate)
    history = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}

    model_archive = f'clf_record/exp1'
    if os.path.exists('clf_record'):
        # n = max([int(i[-1]) for i in os.listdir('clf_record')]) + 1
        # n = int(os.listdir('runs')[-1][-1]) + 1
        # model_archive = f'clf_record/exp{n}'
        model_archive = f'clf_record/PU31_pertrained_512'
    writer = SummaryWriter(model_archive)
    best_acc = 0
    for i in range(epochs):
        print(f'Epoch{i + 1}:-------------------------------')
        train_loss, train_accuracy = train_epoch(hsiclf_15p_204c_sstiny_model, train_dataloader, optimizer, loss_fn)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        writer.add_scalar('Loss/train', sum(train_loss) / len(train_loss), i)
        writer.add_scalar('Acc/train', train_accuracy, i)
        if (i + 1) % 5 == 0:
            test_loss, test_accuracy = test_epoch(hsiclf_15p_204c_sstiny_model, test_dataloader, loss_fn)
            history['test_loss'].append(test_loss)
            history['test_accuracy'].append(test_accuracy)
            writer.add_scalar('Loss/valid', sum(test_loss) / len(test_loss), i)
            writer.add_scalar('Acc/valid', test_accuracy, i)
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                torch.save(hsiclf_15p_204c_sstiny_model.state_dict(), os.path.join(model_archive, 'best.pth'))

        torch.save(hsiclf_15p_204c_sstiny_model.state_dict(), os.path.join(model_archive, 'last.pth'))

    a = 0
