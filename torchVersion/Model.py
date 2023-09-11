from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import pairPredict, contrastLoss
import info_nce as contrastive
from tqdm import tqdm

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.gcnLayer = GCNLayer()
		self.hgnnLayer = HGNNLayer()
		self.uHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
		self.iHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))

		self.edgeDropper = SpAdjDropEdge()
		self.loss = contrastive.InfoNCE(temperature=args.temp, reduction='mean')

	def forward(self, adj, keepRate):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		uuHyper = self.uEmbeds @ self.uHyper
		iiHyper = self.iEmbeds @ self.iHyper

		for i in range(args.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
			hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-keepRate), lats[-1][:args.user])
			hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1-keepRate), lats[-1][args.user:])
			gnnLats.append(temEmbeds)
			hyperLats.append(t.concat([hyperULat, hyperILat], dim=0))
			lats.append(temEmbeds + hyperLats[-1])
		embeds = sum(lats)
		return embeds, gnnLats, hyperLats

	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]

		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40

		users = t.unique(ancs)
		items = t.unique(poss)
		sslLoss = 0
		sslLoss2 = 0

		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds2 = hyperEmbedsLst[i]
			# sslLoss += contrastLoss(embeds1[:args.user], embeds2[:args.user], t.unique(ancs), args.temp) + contrastLoss(embeds1[args.user:], embeds2[args.user:], t.unique(poss), args.temp)
			sslLoss += self.loss(embeds1[users], embeds2[users]) + self.loss(embeds2[args.user+items], embeds2[args.user+items])

		return bprLoss, sslLoss

	def predict(self, adj):
		embeds, _, _= self.forward(adj, 1.0)
		return embeds[:args.user], embeds[args.user:]

class SocialModel(Model):
	def __init__(self,socialAdj):
		super(SocialModel, self).__init__()
		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.gcnLayer = GCNLayer()
		self.sgnnLayer = GCNLayer()
		self.socialAdj = socialAdj
		self.loss = contrastive.InfoNCE(temperature=args.temp, reduction='mean')

		self.edgeDropper = SpAdjDropEdge()

	def forward(self, adj, keepRate):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		for i in range(args.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
			hyperULat = self.sgnnLayer(self.edgeDropper(self.socialAdj, keepRate), lats[-1][:args.user])
			gnnLats.append(temEmbeds)
			hyperLats.append(hyperULat)
			a = temEmbeds[:args.user] + hyperULat
			lats.append(t.concat([a,temEmbeds[args.user:]]))
		embeds = sum(lats)
		return embeds, gnnLats, hyperLats
	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]

		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40
		users = t.unique(ancs)
		items = t.unique(poss)
		sslLoss = 0
		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds2 = hyperEmbedsLst[i]
			sslLoss += self.loss(embeds1[users], embeds2[users])
		return bprLoss, sslLoss
class SocialModel4(Model):
	def __init__(self, tagAdj):
		super(SocialModel4, self).__init__()
		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.tEmbeds = nn.Parameter(init(t.empty(args.tag, args.latdim)))
		self.gcnLayer = GCNLayer()
		self.tgnnLayer = GCNLayer()
		self.tagAdj = tagAdj
		self.loss = contrastive.InfoNCE(temperature=args.temp, reduction='mean')

		self.edgeDropper = SpAdjDropEdge()

	def forward(self, adj, keepRate):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		tagembeds = t.concat([self.iEmbeds, self.tEmbeds], dim=0)
		lats = [embeds]
		gnnLats = []
		tagLats = [tagembeds]

		for i in range(args.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
			tagLat = self.tgnnLayer(self.edgeDropper(self.tagAdj, keepRate), \
									t.concat([lats[-1][args.user:],tagLats[-1][args.item:]]))
			gnnLats.append(temEmbeds)
			tagLats.append(tagLat)
			a = temEmbeds[:args.user]
			b = temEmbeds[args.user:] + tagLat[:args.item]
			lats.append(t.concat([a,b]))
		embeds = sum(lats)
		return embeds, gnnLats, tagLats

	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, gcnEmbedsLst, tagEmbedsLst = self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]

		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40

		users = t.unique(ancs)
		items = t.unique(poss)
		sslLoss = 0
		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds2 = tagEmbedsLst[i+1]
			sslLoss += self.loss(embeds1[args.user + items], embeds2[items])
		return bprLoss, sslLoss

	def predict(self, adj):
		embeds, _, _= self.forward(adj, 1.0)
		return embeds[:args.user], embeds[args.user:]


class SocialModel2(nn.Module):
	def __init__(self,socialAdj, tagAdj):
		super(SocialModel2, self).__init__()
		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.tEmbeds = nn.Parameter(init(t.empty(args.tag, args.latdim)))
		self.gcnLayer = GCNLayer()
		self.tgnnLayer = GCNLayer()
		self.sgnnLayer = GCNLayer()
		self.socialAdj = socialAdj
		self.tagAdj = tagAdj
		self.loss = contrastive.InfoNCE(temperature=args.temp, reduction='mean')

		self.edgeDropper = SpAdjDropEdge()

	def forward(self, adj, keepRate):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		tagembeds = t.concat([self.iEmbeds, self.tEmbeds], dim=0)
		lats = [embeds]
		gnnLats = []
		tagLats = [tagembeds]
		socialLats = []

		for i in range(args.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
			tagLat = self.tgnnLayer(self.edgeDropper(self.tagAdj, keepRate), \
									t.concat([lats[-1][args.user:],tagLats[-1][args.item:]]))
			socialULat = self.sgnnLayer(self.edgeDropper(self.socialAdj, keepRate), lats[-1][:args.user])
			gnnLats.append(temEmbeds)
			tagLats.append(tagLat)
			socialLats.append(socialULat)
			a = temEmbeds[:args.user] + socialULat
			b = temEmbeds[args.user:] + tagLat[:args.item]
			lats.append(t.concat([a,b]))
		embeds = sum(lats)
		return embeds, gnnLats, tagLats, socialLats

	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, gcnEmbedsLst, tagEmbedsLst, socialEmbedsLst = self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]

		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40

		users = t.unique(ancs)
		items = t.unique(poss)
		sslLoss = 0
		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds2 = tagEmbedsLst[i+1]
			embeds3 = socialEmbedsLst[i]
			sslLoss += self.loss(embeds1[users], embeds3[users]) + self.loss(embeds1[args.user + items], embeds2[items])
		return bprLoss, sslLoss

	def predict(self, adj):
		embeds, _, _, _ = self.forward(adj, 1.0)
		return embeds[:args.user], embeds[args.user:]


class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()
		self.act = nn.LeakyReLU(negative_slope=args.leaky)
		# self.se = SELayer(channel=args.latdim, reduction=8)

	def forward(self, adj, embeds):
		# identity = embeds
		out = self.act(t.spmm(adj, embeds))
		# out = t.spmm(adj, embeds)

		# out = self.se(out)
		# out += identity
		# return self.act(out)
		return out

class HGNNLayer(nn.Module):
	def __init__(self):
		super(HGNNLayer, self).__init__()
		self.act = nn.LeakyReLU(negative_slope=args.leaky)

	def forward(self, adj, embeds):
		lat = self.act(adj.T @ embeds)
		ret = self.act(adj @ lat)
		return ret

class SpAdjDropEdge(nn.Module):
	def __init__(self):
		super(SpAdjDropEdge, self).__init__()

	def forward(self, adj, keepRate):
		if keepRate == 1.0:
			return adj
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
		newVals = vals[mask] / keepRate
		newIdxs = idxs[:, mask]
		return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)

class SELayer(nn.Module):
	def __init__(self, channel, reduction=4):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool1d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c = x.size()
		y = self.avg_pool(x).expand_as(x)
		y = self.fc(y).view(b, c)
		return x * y.expand_as(x)
