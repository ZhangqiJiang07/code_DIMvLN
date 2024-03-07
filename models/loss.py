"""Loss"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
	def __init__(self, method, views, alpha=1.0, lamb1=1.0, lamb2=1.0):
		"""
		Parameters:
		-----------
		views: int
			View nunmber of dataset.
		alpha: float
			The balance factor in instance-level loss.
		lamb1: float
			The balance factor of dual prediction loss.
		lamb2: float
			The balance factor of reconstruction loss.
		"""
		super(Loss, self).__init__()
		self.alpha = alpha
		self.lamb1 = lamb1
		self.lamb2 = lamb2
		self.method = method
		self.views = views



	def reconstruction_loss(self, x_dic, x_hat_dic):
		"""Within-view Reconstruction loss
		Parameters:
		-----------
		x_dic: dictionary, float tensor, $\tilde{X}^{(v)}$: (sample_num, feature_dim_v).
			Rebuilding input dictionary {$\tilde{X}^{(1)}$, $\tilde{X}^{(2)}$, ...,
										$\tilde{X}^{(V)}$}.

		x_hat_dic: dictionary, float tensor, $D(E(\tilde{X}^{(v)}))$: (sample_num, feature_dim_v).
			Reconstructed input dictionary {$D(E(\tilde{X}^{(1)}$)), $D(E(\tilde{X}^{(2)}))$, ..., 
										$D(E(\tilde{X}^{(V)}))$}.
		"""
		loss = torch.tensor(0.0, requires_grad=True).cuda()
		for v in range(1, self.views + 1):
			loss += F.mse_loss(x_hat_dic[f'x{v}_hat'], x_dic[f'x{v}'], reduction='mean')

		return loss / self.views



	def dual_prediction_loss(self, z_dic, z_hat_dic):
		"""Cross-view Dual prediction loss
		Parameters:
		-----------
		z_dic: dictionary, float tensor
			The low dimensional representation of the input data.
		z_hat_dic: dictionary, float tensor
			Reconstructed representation.
		"""
		loss = torch.tensor(0.0, requires_grad=True).cuda()
		for v in range(1, self.views + 1):
			loss += F.mse_loss(z_hat_dic[f'z{v}_hat'], z_dic[f'z{v}'], reduction='sum')

		return loss / self.views



	def compute_joint_probability(self, z1, z2):
		""" Compute Joint Probability P_{i,j}
		Parameters:
		-----------
		z1: [sample_num, latent_dim], float tensor
			The representation of input1.
		z2: [sample_num, latent_dim], float tensor
			The representation of input2.

		Returns:
		--------
		P_i_j: [latent_dim, latent_dim], float_tensor
			The joint probability.
		"""
		bn, k = z1.size()
		assert(z2.size()[0] == bn and z2.size()[1] == k)
		P_i_j = z1.unsqueeze(2) * z2.unsqueeze(1) # (bn, k, k)
		P_i_j = P_i_j.sum(dim=0) # (k, k)

		# Symmetrize P(z1, z2) = P(z2, z1)
		P_i_j = (P_i_j + P_i_j.t()) / 2

		# Normalize
		P_i_j = P_i_j / P_i_j.sum()
		
		return P_i_j



	def instance_contrastive_loss(self, z1, z2, alpha=1.0, EPS=sys.float_info.epsilon):
		"""Instance-level contrastive loss
		Parameters:
		-----------
		z1 / z2: Please see 'compute_joint_probability'.
		alpha: float
			Balance factor.
		EPS: Minimum machine accuracy
		"""
		_, k = z1.size()
		# Joint PDF
		P_i_j = self.compute_joint_probability(F.softmax(z1, dim=1), F.softmax(z2, dim=1))

		# Marginal PDF
		P_i = P_i_j.sum(dim=1).view(k, 1)
		P_j = P_i_j.sum(dim=0).view(1, k)

		# Avoid collapse (avoid nan)
		P_i_j = torch.where(P_i_j < EPS, torch.tensor([EPS], device=P_i_j.device), P_i_j)
		P_i = torch.where(P_i < EPS, torch.tensor([EPS], device=P_i.device), P_i)
		P_j = torch.where(P_j < EPS, torch.tensor([EPS], device=P_j.device), P_j)

		loss = - P_i_j * (P_i_j.log() - alpha * P_i.log() - alpha * P_j.log())

		loss = loss.mean()

		return loss



	def cross_instance_contrastive_loss(self, z_dic, alpha=1.0, EPS=sys.float_info.epsilon, device=None):
		loss = torch.tensor(0.0, requires_grad=True).to(device)
		for v in range(1, self.views):
			for j in range(v + 1, self.views + 1):
				loss += self.instance_contrastive_loss(z_dic[f'z{v}'], z_dic[f'z{j}'], alpha=alpha, EPS=EPS)

		return loss



	def category_contrastive_loss(self, concat_z, gt_label, label_mask, classes, device=None):
		"""Category-level contrastive loss
		Parameters:
		-----------
		concat_z: [sample_num, latent_dim + latent_dim], float tensor
			concat_z = [z_1; z_2], where the operator ';' is concatenate.
		gt_label: [sample_num, 1], float tensor
			The real labels of sample.
		label_mask: long tensor, (N,)
			The mask of labels.
		classes: int tensor
			Class number.
		"""

		"""Let:
			concat_z: (N, D);
			gt_label: (N, 1);
			classes: C.
		"""

		labeled_idx = label_mask[:, 0] == 1

		bn = gt_label[labeled_idx].size()[0] # N'
		# Dot product similarity matrix
		con_z = F.softmax(concat_z[labeled_idx], dim=1)

		# S = torch.matmul(concat_z[labeled_idx], concat_z[labeled_idx].t()) # (N', N')
		S = torch.matmul(con_z, con_z.t()) # (N', N')
		# Get diagnoal elements
		S_diag = torch.diag(S)
		S = S - torch.diag_embed(S_diag)

		# Onehot code
		label_onehot = F.one_hot(gt_label[labeled_idx], classes).float() # (N', C)

		# Count the number of samples in each class
		each_class_num = torch.sum(label_onehot, 0, keepdim=True) # (N', C)
		# Calculate the similarity between instances and each class
		S_sum = torch.matmul(S, label_onehot) # (N', C)
		# Construct propagation matrix for summation
		each_class_num_broadcast = each_class_num.repeat([bn, 1]) - label_onehot # (N', C)
		# Avoid zero deviation
		each_class_num_broadcast[each_class_num_broadcast == 0] = 1
		# torch.div element-level deviation
		S_mean = torch.div(S_sum, each_class_num_broadcast) # (N', C)

		# assign labels
		_label = torch.argmax(S_mean, dim=1) # begin from 0

		# E_{Z~T(y)}[S(Z, Z_{t})]
		E_y = torch.max(S_mean, dim=1)[0]

		gamma = (gt_label[labeled_idx] == _label).float() # 1 if assigned label is correct else 0

		# E_{Z~T(gt)}[S(Z, Z_{t})]
		E_gt = S_mean.mul(label_onehot) # element-wise product
		E_gt = torch.sum(E_gt, dim=1)

		loss = torch.mean(torch.relu(torch.add(gamma, torch.sub(E_y, E_gt))))

		return loss



	def classification_loss(self, prob, gt_label, label_mask, device=None, EPS=sys.float_info.epsilon):
		""" Entropy loss for multi-classes.
		Parameters:
		-----------
		prob: float tensor, (N, classes)
			The probability matrix.
		gt_label: long tensor, (N, )
			The ground truth labels of samples.
		label_mask: long tensor, (N, )
			The indicator matrix of whether the label is masked.
		"""
		labeled_idx = label_mask[:, 0] == 1

		prob = torch.where(prob < EPS, torch.tensor([EPS], device=prob.device), prob)
		# Cross-Entropy
		log_prob = torch.log(prob[labeled_idx])
		loss = F.nll_loss(log_prob, gt_label[labeled_idx])

		return loss



	def recovery_loss(self, original_input, completed_input, mask, indices_dic, device=None):
		"""Pre-train GCN """
		loss = torch.tensor(0.0, requires_grad=True).to(device)
		for v in range(1, self.views + 1):

			missing_idx_bool = mask[:, v - 1] == 0
			missing_idx = completed_input['sub_idx'][missing_idx_bool].long().to(device)
			coor_idx = torch.tensor(range(mask.shape[0])).to(device)[missing_idx_bool].long()

			for k, idx in enumerate(missing_idx):
				connected_existing_idx = indices_dic[f'indices{v}'][0, :] == idx
				connected_existing_idx = indices_dic[f'indices{v}'][1, :][connected_existing_idx].long()

				missing_x_hat = torch.index_select(completed_input[f'completed_input{v}'], dim=0, index=coor_idx[k])
				connected_x = torch.index_select(original_input[f'view{v}_x'], dim=0, index=connected_existing_idx)

				loss += F.mse_loss(missing_x_hat.expand(connected_x.shape[0], -1), connected_x, reduction='mean')

		return loss / v



	# def KL_divergence_loss(self, prob, mask, indices_dic):
	# 	"""KL Divergence Loss for GCN
	# 	https://arxiv.org/pdf/2112.00739 formula(11)
	# 	"""
	# 	loss = torch.tensor(0.0).cuda()
	# 	for v in range(1, len(indices_dic)+1):
	# 		missing_idx = mask[:, v - 1] == 0
	# 		missing_idx = torch.tensor(range(missing_idx.shape[0]))[missing_idx].long()
	# 		for idx in missing_idx:
	# 			connected_existing_idx = indices_dic[f'indices{v}'][0, :] == idx
	# 			connected_existing_idx = indices_dic[f'indices{v}'][1, :][connected_existing_idx].long()
				
	# 			missing_prob = torch.index_select(prob, dim=0, index=idx).flatten()
	# 			connected_prob = torch.index_select(prob, dim=0, index=connected_existing_idx)

	# 			loss += torch.mul(-torch.div(connected_prob, missing_prob).log(), missing_prob).sum()

	# 	return loss



	def forward(self, output_dic, gt_label, label_mask, classes=2, mode='without_dual', device=None):

		if mode == 'with_dual':
			"""
			icl + ccl + recon_loss + classification_loss
			(+ dual_pre_loss   if completion method is 'no_complete', 'average_complete', or 'random_complete')
			"""
			# Make suitable dictionary for loss computation
			concat_z = torch.FloatTensor().to(device)
			z_dic, z_hat_dic, x_dic, x_hat_dic = {}, {}, {}, {}
			for v in range(1, self.views + 1):
				concat_z = torch.cat([concat_z, output_dic[f'z{v}']], dim=1)
				z_dic[f'z{v}'] = output_dic[f'z{v}']
				z_hat_dic[f'z{v}_hat'] = output_dic[f'z{v}_hat']
				x_dic[f'x{v}'] = output_dic[f'view{v}_x']
				x_hat_dic[f'x{v}_hat'] = output_dic[f'x{v}_recon']

			loss_rc = self.reconstruction_loss(x_dic, x_hat_dic)
			loss_icl = self.lamb1 * self.cross_instance_contrastive_loss(z_dic, self.alpha, device=device)
			loss_ccl = self.lamb1 * self.category_contrastive_loss(concat_z, gt_label, label_mask, classes, device)
			loss_cel = self.lamb2 * self.classification_loss(output_dic['prob'], gt_label, label_mask, device)
			
			loss = loss_rc + loss_icl + loss_ccl + loss_cel

			if self.method['Completion'] in ['no_complete', 'average_complete', 'random_complete']:
				loss += self.lamb1 * self.dual_prediction_loss(z_dic, z_hat_dic)

			return loss, {'L_rc': loss_rc.item(), 'L_icl': loss_icl.item(), 'L_ccl': loss_ccl.item(), 'L_cel': loss_cel.item()}


		elif mode == 'without_dual':
			"""
			icl + ccl + recon_loss + classification_loss
			"""
			concat_z = torch.FloatTensor().to(device)
			z_dic, z_hat_dic, x_dic, x_hat_dic = {}, {}, {}, {}
			for v in range(1, self.views + 1):
				concat_z = torch.cat([concat_z, output_dic[f'z{v}']], dim=1)
				z_dic[f'z{v}'] = output_dic[f'z{v}']
				x_dic[f'x{v}'] = output_dic[f'view{v}_x']
				x_hat_dic[f'x{v}_hat'] = output_dic[f'x{v}_recon']

			loss_rc = self.reconstruction_loss(x_dic, x_hat_dic)
			loss_icl = self.lamb1 * self.cross_instance_contrastive_loss(z_dic, self.alpha, device=device)
			loss_ccl = self.lamb1 * self.category_contrastive_loss(concat_z, gt_label, label_mask, classes, device)
			loss_cel = self.lamb2 * self.classification_loss(output_dic['prob'], gt_label, label_mask, device)

			loss = loss_rc + loss_icl + loss_ccl + loss_cel

			return loss, {'L_rc': loss_rc.item(), 'L_icl': loss_icl.item(), 'L_ccl': loss_ccl.item(), 'L_cel': loss_cel.item()}
		

		elif mode == 'pre_train_gcn':
			return self.recovery_loss(output_dic['original_input'], output_dic['completed_input'], output_dic['completed_input']['sub_mask'], output_dic['indices_dic'], device), 0

		else:
			raise ValueError('Unknown mode %s' % mode)
















































