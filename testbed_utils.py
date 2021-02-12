import sys, os, math
import numpy as np
import time
import zipfile
import cPickle
import PIL.Image
import PIL.ImageFile
import multiprocessing
import collections
import ctypes
import itertools
import signal
import cv2
from indexlib_gpu_cpu import *
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
import random


caffe_root = os.path.join(os.environ['PRODSEARCH'], 'DVRS-caffe')
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
#caffe.set_device(gpu_id)
caffe.set_mode_gpu()


def dictFromTrainTxt(filename):
    fp = open(filename,'r')
    flines = fp.readlines()
    fp.close()
    flines = np.char.strip(flines)
    flines = np.char.split(flines)
    
    d = {}
    for line in flines:
	line[3] = line[3].split('/')[-1]
	if(line[1] in d):
		d[line[1]].append(line[3])
	else:
		d[line[1]] = [line[3]]
    dic = {}
    for k, v in d.iteritems():
	for val in v:
		gts = v[:]
		gts.remove(val)
		dic[val] = gts
    return dic

def dictFromTxt(filename):
    fp = open(filename,'r')
    flines = fp.readlines()
    fp.close()
    flines = np.char.strip(flines)
    flines = np.char.split(flines, '\t')
    
    d = {}
    for line in flines:
        d[line[0]] = line[1:]
    return d


# functions
def load_net(model_deploy_file, model_weight_file ):
    print "Initializing the net with" , model_deploy_file , "+" , model_weight_file
    sys.stdout.flush()
    net = caffe.Net(model_deploy_file, model_weight_file, caffe.TEST)
    #net.blobs['data'].reshape(1,3,224,224) # the batch size can be changed on-the-fly
    #meanPath = caffe.__path__[0] + '/imagenet/ilsvrc_2012_mean.npy'
    mean = np.array([104,117,123])
    #mean = np.load(meanPath).mean(1).mean(1)
    print 'mean:', mean
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', mean ) # bgColor-'#7B7568'
    transformer.set_transpose('data', (2,0,1)) # puts the channel as the first dimention
    # transformer.set_raw_scale('data', 255.0) # we will receive uint8 raw image
    transformer.set_channel_swap('data', (2,1,0))
    return net, transformer

def extract_batch_feature(net, layer_name, batch_data, flatten_features=True):
    input_size = net.blobs['data'].shape[-1] 
    imgs, keys, shapes = zip(*batch_data)
    batch_im = np.vstack( imgs )
    if net.blobs['data'].data.shape[0] != batch_im.shape[0]:
        net.blobs['data'].reshape(batch_im.shape[0],3,input_size,input_size) # the batch size can be changed on-the-fly
    net.blobs['data'].data[...] = batch_im
    net.forward()
    feat = net.blobs[layer_name].data
    if(flatten_features):
        feat_data = feat.reshape( feat.shape[0] , -1 ).copy() # must copy
    else:
        feat_data = feat.copy()
    return ( feat_data, keys, shapes)

# Return Image object of size input_size (smaller if input image is smaller)
def getScaledSquare_pil(im, input_size, bgColor = '#FFFFFF' ):
    im.thumbnail( (input_size, input_size) , PIL.Image.BICUBIC)
    nim = PIL.Image.new( im.mode, ( max(im.size), max(im.size) ), bgColor )
    offset = (max(im.size)-min(im.size))/2
    where = (0,offset) if im.size[0]>im.size[1] else (offset,0)
    nim.paste( im, where )
    return nim

def getScaledSquareCrop_pil(im, input_size, img_size, bgColor = '#FFFFFF' ):
    nim = getScaledSquare_pil(im, img_size, bgColor )
    if(nim.size!=(img_size, img_size)):
        nim = nim.resize((img_size, img_size), PIL.Image.BILINEAR)
    offset = max( ( img_size - input_size ) / 2, 0 )
    crop_size = ( offset, offset, offset+input_size, offset+input_size )
    return nim.crop(crop_size)

def load_image( fname, input_folder, transformer, preprocess, input_size ):
    try:
        im = PIL.Image.open( os.path.join( input_folder, fname ) )
        im = im.convert("RGB")
        org_size = im.size
        if preprocess == 'square':
            im = getScaledSquare_pil(im, input_size)
        elif preprocess.startswith('cropfrom'):
            img_size = int( preprocess.replace('cropfrom','') )
            im = getScaledSquareCrop_pil(im, input_size, img_size)
        elif preprocess.startswith('cropscale'): 
            # cropscaleX: center crops (X)% width and height, and then scale to input_size with white padding on shorter side
            crop_size = float(preprocess.replace('cropscale',''))/100.0
            width, height = int(round(crop_size*im.size[0])), int(round(crop_size*im.size[1]))
            offset_x, offset_y = (im.size[0] - width)/2, (im.size[1] - height)/2
            im = im.crop((offset_x, offset_y, width + offset_x, height + offset_y))
            im = getScaledSquare_pil(im, input_size)
        else:
		print 'No preprocessing selected'
        im = np.array( im ).astype( np.float32, copy=False )
        im = transformer.preprocess('data', im)
        im = im.reshape( (1,) + im.shape )
        return (im, os.path.join(input_folder, fname), org_size)
    except Exception as ex:
        print "load_image: %s '%s'"%( str(ex) , os.path.join( input_folder, fname ) )
    return None

def img_loader_pool(pool, input_folder, transformer, preprocess, input_size, batch_size, list_file = 'No', num_images = None, shuffleData = False):
	timeout = 100
	try:
		workers = collections.deque()
		q_list = []
		if(list_file=='No'):
			for root, dirname, files in os.walk(input_folder):
				q_list.extend([ (root,fil) for fil in files])
		else:
			q_list.extend( map(lambda x: ('/'.join(x.split('/')[:-1]), x.split('/')[-1]) , map(lambda x: input_folder + x.strip().split(' ')[-1], open(list_file,'r').readlines()[1:]) ) )
		if(shuffleData):
			random.shuffle(q_list)
                q_list = q_list[:num_images]
		for fname in q_list:
			workers.append( pool.apply_async( load_image, (fname[1], fname[0], transformer, preprocess, input_size ) ) )
			if len(workers) >= batch_size:
				result = workers.popleft()
				try:
					yield result.get(timeout)
				except multiprocessing.TimeoutError:
					print 'Timeout Error'
					print result
		for result in workers:
			try:
				yield result.get(timeout)
			except multiprocessing.TimeoutError:
				print 'Timeout Error'
				print result
	except KeyboardInterrupt as ex:
		print >>sys.stderr, "Caught", type(ex), "terminating the pool..."
		pool.terminate()
		pool.join()

def img_loader(input_folder, transformer, preprocess, input_size, batch_size):
	q_list = []
        for root, dirname, files in os.walk(input_folder):
            q_list.extend([ (root,fil) for fil in files])
	print "Got qlist"
        for fname in q_list:
            yield load_image(fname[1], fname[0], transformer, preprocess, input_size )

def check_hit(gt_dict, q_name, r_name):
    q_name = q_name.split('/')[-1]
    r_ids = map(lambda x: x.split('/')[-1], r_name)

    ret = False
    for i, r_elm in enumerate(r_ids):
        if r_elm in gt_dict[q_name]:
        	ret = True
    return ret

def average_precision(gt_dict, q_name, r_name):
    # calculating mean average precision at k
    k = len(r_name)
    q_name = q_name.split('/')[-1].split('.')[0]
    r_ids = map(lambda x: x.split('/')[-1].split('.')[0], r_name)

    hits = 0.0
    score = 0.0
    for i, r_elm in enumerate(r_ids):
        if r_elm in gt_dict[q_name] and r_elm not in r_ids[:i]:
        	hits += 1.0
		score += hits/(i+1.0)

    max_correct = min(len(gt_dict[q_name]), k)
    return score/max_correct

