import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
from tqdm import tqdm
from collections import defaultdict

def _convert_sp_mat_to_sp_tensor(X):
	coo = X.tocoo().astype(np.float32)
	row = t.Tensor(coo.row).long()
	col = t.Tensor(coo.col).long()
	index = t.stack([row, col])
	data = t.FloatTensor(coo.data)
	return t.sparse.FloatTensor(index, data, t.Size(coo.shape))

class DataHandler:
	def __init__(self, social=False, item=False):
		if args.data == 'yelp':
			predir = '../Data/yelp/'
		elif args.data == 'amazon':
			predir = '../Data/amazon/'
		elif args.data == 'ml10m':
			predir = '../Data/ml10m/'
		elif args.data == 'sparse_yelp':
			predir = '../Data/sparse_yelp/'
		elif args.data == 'lastfm':
			predir = '../Data/last/'
		elif args.data == 'ciao':
			predir = '../Data/ciao//try/'
		elif args.data == 'try':
			predir = '../Data/try/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'
		self.trustfile = predir + 'trustMat.pkl'
		self.tagfile = predir + 'tagMat.pkl'
		self.isSocial = social
		self.isItem = item

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)
		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()
	def makeTorchAdj3(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.item, args.item))
		b = sp.csr_matrix((args.tag, args.tag))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)
		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()

	def makeTorchAdj2(self, mat):
		adj_mat = mat.tolil()

		rowsum = np.array(adj_mat.sum(axis=1))
		d_inv = np.power(rowsum, -0.5).flatten()
		d_inv[np.isinf(d_inv)] = 0.
		d_mat = sp.diags(d_inv)

		norm_adj = d_mat.dot(adj_mat)
		norm_adj = norm_adj.dot(d_mat)
		norm_adj = norm_adj.tocsr()
		socialGraph = _convert_sp_mat_to_sp_tensor(norm_adj)
		socialGraph = socialGraph.coalesce().cuda()
		return socialGraph


	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		args.user, args.item = trnMat.shape
		self.torchBiAdj = self.makeTorchAdj(trnMat)
		if self.isSocial:
			trustMat = self.loadOneFile(self.trustfile)
			self.torchSocialAdj = self.makeTorchAdj2(trustMat)

		if self.isItem:
			tagMat = self.loadOneFile(self.tagfile)
			args.item, args.tag = tagMat.shape
			self.torchtagAdj = self.makeTorchAdj3(tagMat)

		# trnData = TrnData(trnMat)
		trnData = ProposedTrnDataItem(trnMat, trustMat,tagMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)
	#Burası değiştiriclecek.
	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class ProposedTrnData(data.Dataset):
	def __init__(self, trnMat , trustMat, tagMat):
		self.rows = trnMat.row
		self.cols = trnMat.col
		self.dokmat = trnMat.todok()
		self.socialmat = trustMat.todok()
		self.categorymat = tagMat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)
		self.dic = defaultdict(set)
		self.sosdic= defaultdict(set)

		for us, it in tqdm(dict(self.dokmat).keys()):
			self.dic[us].add(it)

		for us, it in tqdm(dict(self.socialmat).keys()):
			self.sosdic[us].add(it)
			self.sosdic[it].add(us)

		for user in tqdm(self.dic.keys()):
			for friend in self.sosdic[user]:
				self.dic[user].update(self.dic[friend])
	#Burası değiştiriclecek.
	def negSampling(self):
		u = -1
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if iNeg not in self.dic[u]:
					break
			self.negs[i] = iNeg
		print("Negative Sampling is finished! \n")

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class ProposedTrnDataItem(data.Dataset):
	def __init__(self, trnMat , trustMat, tagMat):
		self.rows = trnMat.row
		self.cols = trnMat.col
		self.dokmat = trnMat.todok()
		self.socialmat = trustMat.todok()
		self.categorymat = tagMat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)
		self.dic = defaultdict(set)
		self.catdic = defaultdict(set)
		self.catdicCAT = defaultdict(set)


		for us, it in tqdm(dict(self.dokmat).keys()):
			self.dic[us].add(it)

		for us, it in tqdm(dict(self.categorymat).keys()):
			self.catdic[us].add(it)
			self.catdicCAT[it].add(us)


		for user in tqdm(self.dic.keys()):
			same_items = set()
			for item in self.dic[user]:
				for cat in self.catdic[item]:
					same_items.update(self.catdicCAT[cat])
			self.dic[user].update(same_items)

	#Burası değiştiriclecek.
	def negSampling(self):
		u = -1
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if iNeg not in self.dic[u]:
					break
			self.negs[i] = iNeg
		print("Negative Sampling is finished! \n")

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])



