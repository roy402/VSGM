import torch
from torch import nn
from torch.nn import functional as F


# https://docs.dgl.ai/en/latest/tutorials/models/3_generative_model/5_dgmg.html#dgmg-the-main-flow
class WeightedSum(nn.Module):
    def __init__(self, INPUT_FEATURE_SIZE, EMBED_FEATURE_SIZE, NUM_CHOSE_NODE):
        super(WeightedSum, self).__init__()

        # Setting from the paper
        # 40
        self.node_feature_size = INPUT_FEATURE_SIZE
        # 64
        self.embed_feature_SIZE = EMBED_FEATURE_SIZE
        self.NUM_CHOSE_NODE = NUM_CHOSE_NODE

        # Embed graphs
        ### every nodes weight
        self.node_gating = nn.Sequential(
            nn.Linear(self.node_feature_size, 1),
            nn.Sigmoid()
        )
        self.node_to_graph = nn.Linear(self.node_feature_size,
                                       self.embed_feature_SIZE)

    def forward(self, features):
        '''
        features: torch.Size([10, 40])
        '''
        # import pdb; pdb.set_trace()
        score = self.node_gating(features)
        merge_features = (score * self.node_to_graph(features)).sum(0, keepdim=True)
        indices = torch.argsort(score, dim=0, descending=True)[:self.NUM_CHOSE_NODE]
        dict_ANALYZE_GRAPH = {}
        dict_ANALYZE_GRAPH["score"] = score[indices].view(-1).clone().detach().to('cpu').tolist()
        dict_ANALYZE_GRAPH["sort_nodes_index"] = indices[:self.NUM_CHOSE_NODE+5].view(-1).clone().detach().tolist()
        return merge_features, dict_ANALYZE_GRAPH

class SelfAttn(nn.Module):

    def __init__(self, INPUT_FEATURE_SIZE, EMBED_FEATURE_SIZE, NUM_CHOSE_NODE):
        super().__init__()
        self.NUM_CHOSE_NODE = NUM_CHOSE_NODE
        self.node_to_graph = nn.Linear(INPUT_FEATURE_SIZE,
                                       EMBED_FEATURE_SIZE)
        self.scorer = nn.Linear(INPUT_FEATURE_SIZE, 1)

    def forward(self, features):
        scores = F.softmax(self.scorer(features), dim=1)
        merge_features = scores.bmm(self.node_to_graph(features)).sum(0, keepdim=True)
        indices = torch.argsort(scores, dim=0, descending=True)[:self.NUM_CHOSE_NODE]
        dict_ANALYZE_GRAPH = {}
        dict_ANALYZE_GRAPH["score"] = scores[indices].view(-1).clone().detach().to('cpu').tolist()
        dict_ANALYZE_GRAPH["sort_nodes_index"] = indices[:self.NUM_CHOSE_NODE+5].view(-1).clone().detach().tolist()
        return merge_features


class DotAttn(nn.Module):
    '''
    dot-attention (or soft-attention)
    '''

    def forward(self, inp, h):
        score = self.softmax(inp, h)
        # [2, 145, 1] -> [2, 145, 1024] -> * inp -> sum all 145 word to 1024 feature
        # -> [2, 1024]
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        '''
        inp : [2, 145, 1024]
        h : [2, 1024]
        '''
        # import pdb; pdb.set_trace()
        # [2, 145, 1]
        raw_score = inp.bmm(h.unsqueeze(2))
        # [2, 145, 1]
        score = F.softmax(raw_score, dim=1)
        return score


class DotAttnChoseImportentNode(nn.Module):
    def __init__(self, bert_hidden_size, NODE_FEATURE_SIZE, NUM_CHOSE_NODE, GPU, PRINT_DEBUG):
        super().__init__()
        self.NUM_CHOSE_NODE = NUM_CHOSE_NODE
        self.bert_hidden_size = bert_hidden_size
        self.OUTPUT_SHAPE = NODE_FEATURE_SIZE * NUM_CHOSE_NODE
        self.GPU = GPU
        self.PRINT_DEBUG = PRINT_DEBUG
        # 64 -> 40
        self.hidden_state_to_node = nn.Linear(bert_hidden_size, NODE_FEATURE_SIZE)

    def forward(self, nodes, hidden_state):
        '''
        nodes: torch.Size([3, 40])
        hidden_state: torch.Size([1, 64])
        '''
        # tensor([])
        if hidden_state is None or len(hidden_state) == 0:
            if self.PRINT_DEBUG:
                print("WARNING hidden_state is None")
            hidden_state = torch.zeros((1, self.bert_hidden_size))
            if self.GPU:
                hidden_state = hidden_state.to(self.hidden_state_to_node.device)
        elif hidden_state.shape[0] > 1:
            raise "nodes only can accept one batch, hidden_state == torch.Size([1, ?])"
        # torch.Size([1, 40])
        hidden_state = self.hidden_state_to_node(hidden_state)
        # torch.Size([3])
        score = self.softmax(nodes, hidden_state)

        '''
        choose nodes
        '''
        weight_nodes = score.expand_as(nodes).mul(nodes)
        indices = torch.argsort(score, dim=0, descending=True)[:self.NUM_CHOSE_NODE]
        chose_nodes = weight_nodes[indices].view(1, -1)

        # can chose nodes smaller than OUTPUT_SHAPE, cat zeros vectors
        if chose_nodes is None or self.OUTPUT_SHAPE != chose_nodes.shape[-1]:
            tensor_zeros = torch.zeros((chose_nodes.shape[0], self.OUTPUT_SHAPE - chose_nodes.shape[-1]))
            if self.GPU:
                tensor_zeros = tensor_zeros.to('cuda')
            chose_nodes = torch.cat((chose_nodes, tensor_zeros), dim=1)
        dict_ANALYZE_GRAPH = {}
        dict_ANALYZE_GRAPH["score"] = score[indices].view(-1).clone().detach().to('cpu').tolist()
        dict_ANALYZE_GRAPH["sort_nodes_index"] = indices[:self.NUM_CHOSE_NODE+5].view(-1).clone().detach().tolist()
        return chose_nodes, dict_ANALYZE_GRAPH

    def softmax(self, nodes, hidden_state):
        '''
        nodes: torch.Size([3, 40])
        hidden_state : torch.Size([1, 40])
        '''
        # nodes * hidden_state -> torch.Size([3, 1])
        score = torch.matmul(nodes, hidden_state.T)
        score = F.softmax(score, dim=0)
        return score


class WeightedSoftmaxSum(nn.Module):
    def __init__(self, bert_hidden_size, NODE_FEATURE_SIZE, EMBED_FEATURE_SIZE, NUM_CHOSE_NODE, GPU, PRINT_DEBUG):
        super().__init__()
        self.NUM_CHOSE_NODE = NUM_CHOSE_NODE
        self.bert_hidden_size = bert_hidden_size
        self.OUTPUT_SHAPE = NODE_FEATURE_SIZE * NUM_CHOSE_NODE
        self.GPU = GPU
        self.PRINT_DEBUG = PRINT_DEBUG
        # 64 -> 40
        self.hidden_state_to_node = nn.Linear(bert_hidden_size, NODE_FEATURE_SIZE)
        # embed
        self.node_to_graph = nn.Linear(NODE_FEATURE_SIZE,
                                       EMBED_FEATURE_SIZE)

    def forward(self, nodes, hidden_state):
        '''
        nodes: torch.Size([3, 40])
        hidden_state: torch.Size([1, 64])
        '''
        # tensor([])
        if hidden_state is None or len(hidden_state) == 0:
            if self.PRINT_DEBUG:
                print("WARNING hidden_state is None")
            hidden_state = torch.zeros((1, self.bert_hidden_size))
            if self.GPU:
                hidden_state = hidden_state.to(self.hidden_state_to_node.device)
        elif hidden_state.shape[0] > 1:
            raise "nodes only can accept one batch, hidden_state == torch.Size([1, ?])"
        # torch.Size([1, 40])
        hidden_state = self.hidden_state_to_node(hidden_state)
        # torch.Size([3])
        score = self.softmax(nodes, hidden_state)

        merge_features = (score * self.node_to_graph(nodes)).sum(0, keepdim=True)
        indices = torch.argsort(score, dim=0, descending=True)[:self.NUM_CHOSE_NODE]

        dict_ANALYZE_GRAPH = {}
        dict_ANALYZE_GRAPH["score"] = score[indices].view(-1).clone().detach().to('cpu').tolist()
        dict_ANALYZE_GRAPH["sort_nodes_index"] = indices[:self.NUM_CHOSE_NODE+5].view(-1).clone().detach().tolist()
        return merge_features, dict_ANALYZE_GRAPH

    def softmax(self, nodes, hidden_state):
        '''
        nodes: torch.Size([3, 40])
        hidden_state : torch.Size([1, 40])
        '''
        # nodes * hidden_state -> torch.Size([3, 1])
        score = torch.matmul(nodes, hidden_state.T)
        score = F.softmax(score, dim=0)
        return score
