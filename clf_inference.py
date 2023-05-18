import torch
from torch import nn
from common.datautils import *
from models.clf import Classfier

if __name__ == '__main__':
    # imp = r'../data/Salinas_corrected.mat'
    # gtp = r'../data/Salinas_gt.mat'
    # imp = r'../data/Indian_pines_corrected.mat'
    # gtp = r'../data/Indian_pines_gt.mat'
    imp = r'../data/PaviaU.mat'
    gtp = r'../data/PaviaU_gt.mat'

    dataset = HSIloader(img_path=imp, gt_path=gtp, patch_size=25, sample_mode='ratio', train_ratio=0.1,
                        sample_points=30, merge=None, rmbg=False)
    dataset(spectral=False)

    x_test_patch = dataset.x_test_patch.copy()
    del dataset.x_test_patch
    x_train_patch = dataset.x_train_patch.copy()
    del dataset.x_train_patch
    patchss = np.concatenate([x_test_patch, x_train_patch])
    del x_train_patch
    del x_test_patch
    class_num = 9
    model = Classfier(name='hsimae_15p_204c_stiny_model', in_chans=103, class_num=class_num,
                      encoder_embed_dim=256, encoder_depth=4, encoder_num_heads=8,
                      mlp_ratio=4., norm_layer=nn.LayerNorm)
    test_dataset = HSIDataset(patchss, dataset.gt,
                              np.concatenate([dataset.coordinate_test, dataset.coordinate_train]))

    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)
    clssss = {}
    for i in range(dataset.gt.max()):
        clssss[i + 1] = np.where(dataset.gt == i + 1)

    order_dict = torch.load(r'clf_record/PU25_pertrained_512/best.pth')
    model.load_state_dict(order_dict, strict=True)
    if torch.cuda.is_available():
        model.cuda()
    pred = np.empty(shape=(0,))
    labels = np.empty(shape=(0,))
    result = np.zeros_like(dataset.gt)
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            pred = np.concatenate([pred, model(X.cuda()).argmax(1).cpu().numpy()])
            labels = np.concatenate([labels, y.cpu().numpy()])
    pred = pred.astype('int')
    labels = labels.astype('int')
    cls_nums = {}
    for i in range(labels.max()):
        cls_nums[i] = sum(labels == i)
    from sklearn.metrics import confusion_matrix

    CM = confusion_matrix(labels, pred)
    aa = []
    for i in range(labels.max()):
        aa.append(CM[i, i] / cls_nums[i])
    AA = sum(aa) / len(aa)

    l = pred - labels
    OA = sum(l == 0) / len(l)

    k = 0
    for i in range(labels.max()):
        c = sum(CM[i, :])
        r = sum(CM[:, i])
        k = k + r * c
    k = k / (len(pred) ** 2)
    Kappa = (OA - k) / (1 - k)
    print(f'OA={OA},\nAA={AA},\nKappa={Kappa}')

    coordinate = np.concatenate([dataset.coordinate_test, dataset.coordinate_train])
    for i, v in enumerate(coordinate):
        result[v[0], v[1]] = pred[i]+1

    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(result, cmap='jet')
    plt.title('prediction')
    plt.subplot(122)
    plt.imshow(dataset.gt, cmap='jet')
    plt.title('ground_truth')
    plt.show()
    a = 0
