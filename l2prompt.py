import torch
import torch.nn as nn

'''
假设我们有以下参数和输入：
length=5：提示的长度
embed_dim=768：嵌入维度
prompt_pool=True：使用提示池
pool_size=10：提示池大小
top_k=3：选择的提示数量
batch_size=4：批次大小
sequence_length=128：序列长度


输入嵌入：(4, 128, 768)
嵌入均值：(4, 768)
归一化后的提示键：(10, 768)
归一化后的嵌入：(4, 768)
相似度矩阵：(4, 10)
选定的索引：(4, 3)
选定的提示（原始）：(4, 3, 5, 768)
选定的提示（重新组织后）：(4, 15, 768)
带有提示的嵌入：(4, 143, 768)
'''


class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform', proj = False):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        # 添加prompt投影层
        self.proj = proj

        '提示池  一个提示由length个token组成'
        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

            if self.proj:
                self.prompt_proj = nn.Linear(embed_dim, embed_dim)
                nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
                # self.prompt_proj = nn.Sequential(
                #     nn.Linear(embed_dim, embed_dim*4),
                #     nn.Tanh(),
                #     nn.Linear(embed_dim*4, embed_dim))
                # nn.init.kaiming_normal_(self.prompt_proj[0].weight, a=0, mode='fan_out')
                # nn.init.kaiming_normal_(self.prompt_proj[2].weight, a=0, mode='fan_out')

                # self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
                self.prompt_dropout = nn.Dropout(0.1)

        '''
        初始化提示键 nn.Parameter
        key的shape就是单独的一个embedding
        '''
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            # 本质对length维度上的数据降维取平均
            # （5，10，3） -> (5 , 3)
            # 对10个length的每个元素（3个）求平均
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    'x_embed就是图像的token, cls_feature是经过一个完整ViT模型后，进入最终分类器之前的embedding，论文中是q(x)'
    '这个模块的作用，传入图像的embed数据，然后根据这个数据从pool中选择出最适合的prompt，最后将其拼接起来并返回，这里同时计算了key的损失函数'

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            '处理图像嵌入'
            # 根据指定的方法计算嵌入键
            # 这里传入的x_embed是图像的token ( batch, token数量 ,dim)
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            # 归一化提示键和嵌入
            # l2归一化： 对embed_dim维度的数据进行归一化，使得向量长度为1
            # [[1,2,3],**]  -> [1/sqrt(14), 2/sqrt(14), 3/sqrt(14),**]
            # 换种理解： 假设2维度，dim=0 则竖着， dim=1 则横着
            prompt_norm = self.l2_normalize(self.prompt_key, dim=1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C

            '计算相似度'
            # 画图理解
            # 二维矩阵每i行代表，第i个key与所有img的相似程度。
            # 每j个列代表，第j个图像与所有key的相似度
            # 所以这里选择图像时要使用j列来选择出最适合的key
            similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size

            '选择最相似的提示 这里获得id'
            if prompt_mask is None:
                # dim就是竖着处理  挑出最大的前k个
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
                # 根据批量共现频率选择的最相关提示键索引。
                # 也就是说之前是使用单独图像的提示，现在使用batch个图像共同决定使用哪些key
                # 我理解的是因为batch一般是同一个任务？？？
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],),
                                                                     torch.min(idx.flatten()),
                                                                     device=prompt_id.device)])
                        id_counts = torch.cat(
                            [id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                    major_prompt_id = prompt_id[major_idx]  # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k
            else:
                idx = prompt_mask  # B, top_k

            '获取选定的提示  得到数据：每一个图像 有top_k*length 个 embedding / token'
            # prompt转换为batch形式，方便后面传入模型
            # pool_size -> top_k 然后多个一个batch维度
            batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            # too_k和lenght本质都是embedding的数量，将它们合并
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)  # B, top_k * length, C

            # 对prompt进行投影，残差连接
            if self.proj:
                batched_prompt = self.prompt_proj(batched_prompt) + batched_prompt

                batched_prompt = self.prompt_dropout(batched_prompt)

            # out['prompt_idx'] = idx

            # Debugging, return sim as well
            # out['prompt_norm'] = prompt_norm
            # out['x_embed_norm'] = x_embed_norm
            # out['similarity'] = similarity

            "计算损失  这里的损失是计算的图像embedding和key的"
            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx]  # B, top_k, C
            # out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C  余弦相似度
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            # 如果不使用提示池
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        # 返回带有提示的嵌入  这个类主要就是在pool中选取了指定的prompt，然后concat起来并返回。
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        # out['total_prompt_len'] = batched_prompt.shape[1]
        # out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        out['prompted_embedding'] = torch.cat((
            x_embed[:, :1, :],
            batched_prompt,
            x_embed[:, 1:, :]
        ), dim=1)

        return out