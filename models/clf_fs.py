import torch.nn.functional as F
from torch import nn
from timm.models.vision_transformer import Block
from models.PixelEmbed import PixelEmbed, PosCNN
from models.LinearComb import Linear_comb
from common.datautils import *
from torch.utils.tensorboard import SummaryWriter


class Classifier_fs(nn.Module):
    def __init__(self, name=None, in_chans=None, class_num=10,
                 encoder_embed_dim=512, encoder_depth=16, encoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.name = name
        # TODO: encoder
        self.patch_embed = PixelEmbed(in_channels=in_chans, embed_dim=encoder_embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.pos_embed = PosCNN(in_chans=encoder_embed_dim, embed_dim=encoder_embed_dim)

        self.blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.norm = norm_layer(encoder_embed_dim)
        # --------------------------------------------------------------------------------------------------------------

        self.linear_comb = Linear_comb(embed_dim=encoder_embed_dim)
        # TODO: clf partition
        self.fc = nn.Sequential(
            nn.Linear(encoder_embed_dim, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.head = nn.Linear(1024, class_num)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.initialize_weights()

    def initialize_weights(self):

        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear, nn.Conv and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            w = m.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            w = m.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if isinstance(m, nn.Conv3d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encoder_forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def clf(self, x):
        latent = self.fc(x)
        pred = self.head(latent)
        return pred

    def forward(self, x):
        latent = self.encoder_forward(x)
        token = self.linear_comb(latent)

        # logits = self.relu(logits)
        logits = self.clf(token)
        # logits = self.head(logits)
        # logits = self.softmax(logits)
        return token, logits


def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if metric == 'dot':
        logits = torch.mm(feat, proto.t())
    elif metric == 'cos':
        logits = torch.mm(F.normalize(feat, dim=-1),
                          F.normalize(proto, dim=-1).t())
    elif metric == 'sqr':
        logits = -(feat.unsqueeze(1) -
                   proto.unsqueeze(0)).pow(2).sum(dim=-1)

    return logits * temp


def Shannon_entropy_loss(logits):
    entropy = 0
    for item in logits:
        entropy += sum(-i*torch.log(i) for i in item)
    return entropy/len(logits)


if __name__ == '__main__':
    # imp = r'../data/Salinas_corrected.mat'
    # gtp = r'../data/Salinas_gt.mat'
    # imp = r'../data/Indian_pines_corrected.mat'
    # gtp = r'../data/Indian_pines_gt.mat'
    imp = r'data/PaviaU.mat'
    gtp = r'data/PaviaU_gt.mat'
    class_num = 9
    learning_rate = 1e-3
    epochs = 300
    batch_size = 16
    pretrain = True
    SPECTRAL = False
    num_per_class_support = 5
    num_per_class_query = 20

    dataset = HSIloader(img_path=imp, gt_path=gtp, patch_size=15, sample_mode='few-shot', shots=5,
                        query_nums=20, merge=None, rmbg=False)
    dataset(spectral=SPECTRAL)
    train_dataset = HSIDataset(dataset.train_set, dataset.gt, dataset.train_set_coordinate, transform=None)
    test_dataset = HSIDataset(dataset.test_set, dataset.gt, dataset.test_set_coordinate, transform=None)
    support_set_da = dataset.support_set_da
    query_set = dataset.query_set

    hsiclf_15p_204c_stiny_model = Classifier_fs(name='hsimae_15p_204c_stiny_model', in_chans=103, class_num=class_num,
                                                encoder_embed_dim=256, encoder_depth=4, encoder_num_heads=16,
                                                mlp_ratio=4., norm_layer=nn.LayerNorm)

    # support_dataloader, query_dataloader = meta_task(support_set_da, query_set, ways=9, shots=5, queries=20,
    #                                                  num_per_class_support=1, num_per_class_query=10, episode=10)


    # load masked autoencoder pretrain
    if pretrain:
        pretrain_dict = torch.load(r'../runs/exp1/best.pth')
        clf_dict = hsiclf_15p_204c_stiny_model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items()
                         if (k in clf_dict) and (k not in ['patch_embed.conv2d_1.weight'])}
        # hsiclf_15p_204c_stiny_model.load_state_dict(torch.load(r'../runs/exp1/best.pth'), strict=False)
        clf_dict.update(pretrain_dict)
        hsiclf_15p_204c_stiny_model.load_state_dict(clf_dict)
    # check device. move model to GPU
    if torch.cuda.is_available():
        hsiclf_15p_204c_stiny_model.cuda()

    ce_loss_fn = nn.CrossEntropyLoss()
    if pretrain:
        optimizer = torch.optim.AdamW([
            {'params': hsiclf_15p_204c_stiny_model.patch_embed.parameters()},
            {'params': hsiclf_15p_204c_stiny_model.pos_embed.parameters(), 'lr': 1e-5},
            {'params': hsiclf_15p_204c_stiny_model.blocks.parameters(), 'lr': 1e-5},
            {'params': hsiclf_15p_204c_stiny_model.norm.parameters(), 'lr': 1e-5},
            {'params': hsiclf_15p_204c_stiny_model.fc.parameters()},
            {'params': hsiclf_15p_204c_stiny_model.linear_comb.parameters()},
            {'params': hsiclf_15p_204c_stiny_model.head.parameters()},

        ], lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(hsiclf_15p_204c_stiny_model.parameters(), lr=learning_rate)
    history = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}

    for i in range(1000):
        support_dataloader, query_dataloader = meta_task(support_set_da, query_set, ways=9, shots=5, queries=20,
                                                         num_per_class_support=num_per_class_support,
                                                         num_per_class_query=num_per_class_query, episode=i)
        support_datas, support_labels = support_dataloader.__iter__().__next__()

        for (query_datas, query_labels) in query_dataloader:
            support_features, support_logits = hsiclf_15p_204c_stiny_model(support_datas.float().cuda())
            if num_per_class_support > 1:
                support_proto = support_features.reshape(class_num, num_per_class_support, -1).mean(1)
            else:
                support_proto = support_features
            query_features, _ = hsiclf_15p_204c_stiny_model(query_datas.float().cuda())

            logits = F.softmax(compute_logits(query_features, support_proto, metric='cos'), -1)

            loss_entropy = Shannon_entropy_loss(logits)
            CE_loss = ce_loss_fn(support_logits, support_labels.cuda())
            loss = loss_entropy+CE_loss
            optimizer.zero_grad()
            CE_loss.backward()
            optimizer.step()

    model_archive = f'fs_clf_record/exp1'
    if os.path.exists('fs_clf_record'):
        n = max([int(i[-1]) for i in os.listdir('../clf_record')]) + 1
        # n = int(os.listdir('runs')[-1][-1]) + 1
        model_archive = f'clf_record/exp{n}'
    writer = SummaryWriter(model_archive)

    for i in range(epochs):
        pass

    a = 0
