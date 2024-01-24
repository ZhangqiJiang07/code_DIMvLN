""" Config for gcn completion method."""


def make_config_with_gcn_completion(data_name):
	if data_name == 'Caltech101-20':
		return dict(
			seed=4,
			view=2,
			ktop=10,
			Training=dict(
				lr=1.0e-3,
				start_dual_prediction=50,
				pre_train_epoch=20,
				# pre_train_epoch=0,
				recomplete_freq=50,
				batch_size=64,
				epoch=200,
				alpha=10,
				lambda2=1,
				lambda1=1,
			),
			Autoencoder=dict(
				view_size=2,
				arch1=[1984, 1024, 1024, 1024, 128],
				arch2=[512, 1024, 1024, 1024, 128],
				activation1='relu',
				activation2='relu',
				batchnorm=True
				# batchnorm=False
			),
			GCN=dict(
				arch1=[1984, 1024, 1984],
				arch2=[512, 256, 512],
				activation1='relu',
				activation2='relu'
			)
		)

	elif data_name == 'Cub':
		return dict(
			view=2,
			seed=4,
			ktop=10,
			Training=dict(
				lr=1.0e-4,
				start_dual_prediction=50,
				pre_train_epoch=50,
				recomplete_freq=10,
				batch_size=32,
				epoch=200,
				alpha=10,
				lambda2=0.1,
				lambda1=0.1
			),
			Autoencoder=dict(
				arch1=[1024, 1024, 1024, 1024, 128],
				arch2=[300, 1024, 1024, 1024, 128],
				activation1='relu',
				activation2='relu',
				batchnorm=True
			),
			GCN=dict(
				arch1=[1024, 512, 1024],
				arch2=[300, 128, 300],
				activation1='relu',
				activation2='relu'
				)
		)

	elif data_name == 'NUS-WIDE':
		return dict(
			view=5,
			seed=4,
			ktop=12,
			Training=dict(
				lr=1.0e-4,
				start_dual_prediction=50,
				pre_train_epoch=20,
				recomplete_freq=10,
				batch_size=512,
				epoch=200,
				alpha=10,
				lambda2=0.1,
				lambda1=0.1
				),
			Autoencoder=dict(
				arch1=[65, 256, 256, 256, 40],
				arch2=[226, 512, 512, 512, 40],
				arch3=[145, 256, 256, 256, 40],
				arch4=[74, 256, 256, 256, 40],
				arch5=[129, 256, 256, 256, 40],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				activation4='relu',
				activation5='relu',
				batchnorm=True
				),
			GCN=dict(
				arch1=[65, 32, 65],
				arch2=[226, 128, 226],
				arch3=[145, 64, 145],
				arch4=[74, 32, 74],
				arch5=[129, 64, 129],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				activation4='relu',
				activation5='relu',
				)
			)

	elif data_name == 'Wiki':
		return dict(
			view=2,
			seed=4,
			ktop=10,
			Training=dict(
				lr=1.0e-4,
				start_dual_prediction=50,
				pre_train_epoch=200,
				recomplete_freq=10,
				batch_size=32,
				epoch=200,
				alpha=10,
				lambda2=0.1,
				lambda1=0.1
			),
			Autoencoder=dict(
				arch1=[128, 1024, 1024, 1024, 128],
				arch2=[10, 1024, 1024, 1024, 128],
				activation1='relu',
				activation2='relu',
				batchnorm=True
			),
			GCN=dict(
				arch1=[128, 64, 128],
				arch2=[10, 4, 10],
				activation1='relu',
				activation2='relu'
				)
		)

	elif data_name == 'HandWritten':
		return dict(
			view=6,
			seed=4,
			ktop=10,
			Training=dict(
				lr=1.0e-4,
				start_dual_prediction=50,
				pre_train_epoch=100,
				recomplete_freq=10,
				batch_size=256,
				epoch=200,
				alpha=10,
				lambda2=0.1,
				lambda1=0.1
				),
			Autoencoder=dict(
				arch1=[240, 128, 128, 128, 32],
				arch2=[76, 128, 128, 128, 32],
				arch3=[216, 128, 128, 128, 32],
				arch4=[47, 128, 128, 128, 32],
				arch5=[64, 128, 128, 128, 32],
				arch6=[6, 128, 128, 128, 32],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				activation4='relu',
				activation5='relu',
				activation6='relu',
				batchnorm=True
				),
			GCN=dict(
				arch1=[240, 128, 240],
				arch2=[76, 32, 76],
				arch3=[216, 128, 216],
				arch4=[47, 32, 47],
				arch5=[64, 32, 64],
				arch6=[6, 3, 6],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				activation4='relu',
				activation5='relu',
				activation6='relu',
				)
			)

	elif data_name == 'ALOI':
		return dict(
			view=4,
			seed=4,
			ktop=10,
			Training=dict(
				lr=1.0e-4,
				start_dual_prediction=50,
				pre_train_epoch=20,
				recomplete_freq=10,
				batch_size=128,
				epoch=200,
				alpha=10,
				lambda2=0.1,
				lambda1=0.1
				),
			Autoencoder=dict(
				arch1=[77, 512, 512, 512, 40],
				arch2=[13, 512, 512, 512, 40],
				arch3=[64, 512, 512, 512, 40],
				arch4=[125, 512, 512, 512, 40],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				activation4='relu',
				batchnorm=True
				),
			GCN=dict(
				arch1=[77, 32, 77],
				arch2=[13, 32, 13],
				arch3=[64, 64, 64],
				arch4=[125, 64, 125],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				activation4='relu',
				)
			)

	elif data_name == 'Out-Scene':
		return dict(
			view=4,
			seed=4,
			ktop=10,
			Training=dict(
				lr=1.0e-4,
				start_dual_prediction=50,
				pre_train_epoch=20,
				recomplete_freq=10,
				batch_size=128,
				epoch=200,
				alpha=10,
				lambda2=0.1,
				lambda1=0.1
				),
			Autoencoder=dict(
				arch1=[512, 512, 512, 512, 128],
				arch2=[432, 512, 512, 512, 128],
				arch3=[256, 512, 512, 512, 128],
				arch4=[48, 512, 512, 512, 128],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				activation4='relu',
				batchnorm=True
				),
			GCN=dict(
				arch1=[512, 256, 512],
				arch2=[432, 256, 432],
				arch3=[256, 128, 256],
				arch4=[48, 32, 48],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				activation4='relu',
				)
			)

	elif data_name == 'YouTubeFace':
		return dict(
			view=5,
			seed=4,
			ktop=10,
			Training=dict(
				lr=1.0e-4,
				start_dual_prediction=50,
				pre_train_epoch=10,
				recomplete_freq=10,
				batch_size=1024,
				epoch=100,
				alpha=10,
				lambda2=0.1,
				lambda1=0.1
				),
			Autoencoder=dict(
				arch1=[64, 512, 512, 128],
				arch2=[512, 1024, 1024, 128],
				arch3=[64, 512, 512, 128],
				arch4=[647, 1024, 1024, 128],
				arch5=[838, 1024, 1024, 128],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				activation4='relu',
				activation5='relu',
				batchnorm=True
				),
			GCN=dict(
				arch1=[64, 32, 64],
				arch2=[512, 256, 512],
				arch3=[64, 32, 64],
				arch4=[647, 256, 647],
				arch5=[838, 512, 838],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				activation4='relu',
				activation5='relu',
				)
			)

	elif data_name == 'Animal':
		return dict(
			view=4,
			seed=4,
			ktop=10,
			Training=dict(
				lr=1.0e-4,
				start_dual_prediction=50,
				pre_train_epoch=10,
				recomplete_freq=10,
				batch_size=256,
				epoch=280,
				alpha=10,
				lambda2=0.1,
				lambda1=0.1
				),
			Autoencoder=dict(
				arch1=[2689, 1024, 1024, 1024, 128],
				arch2=[2000, 1024, 1024, 1024, 128],
				arch3=[2001, 1024, 1024, 1024, 128],
				arch4=[2000, 1024, 1024, 1024, 128],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				activation4='relu',
				batchnorm=True
				),
			GCN=dict(
				arch1=[2689, 1024, 2689],
				arch2=[2000, 1024, 2000],
				arch3=[2001, 1024, 2001],
				arch4=[2000, 1024, 2000],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				activation4='relu',
				)
			)

	elif data_name == 'STL10':
		return dict(
			view=3,
			seed=4,
			ktop=10,
			Training=dict(
				lr=1.0e-4,
				start_dual_prediction=50,
				pre_train_epoch=10,
				recomplete_freq=10,
				batch_size=128,
				epoch=200,
				alpha=10,
				lambda2=0.1,
				lambda1=0.1
				),
			Autoencoder=dict(
				arch1=[1024, 512, 512, 512, 128],
				arch2=[512, 512, 512, 512, 128],
				arch3=[2048, 512, 512, 512, 128],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				batchnorm=True
				),
			GCN=dict(
				arch1=[1024, 512, 1024],
				arch2=[512, 256, 512],
				arch3=[2048, 1024, 2048],
				activation1='relu',
				activation2='relu',
				activation3='relu',
				)
			)

	elif data_name == 'NoisyMNIST':
		return dict(
			view=2,
			seed=4,
			ktop=10,
			Training=dict(
				lr=1.0e-4,
				start_dual_prediction=50,
				pre_train_epoch=200,
				recomplete_freq=10,
				batch_size=128,
				epoch=100,
				alpha=10,
				lambda2=1,
				lambda1=0.01
				),
			Autoencoder=dict(
				arch1=[784, 1024, 1024, 1024, 128],
				arch2=[784, 1024, 1024, 1024, 128],
				activation1='relu',
				activation2='relu',
				batchnorm=True
				),
			GCN=dict(
				arch1=[784, 512, 256, 512, 784],
				arch2=[784, 512, 256, 512, 784],
				activation1='relu',
				activation2='relu',
				)
			)

	else:
		raise ValueError('Unknown data name %s' % data_name)



