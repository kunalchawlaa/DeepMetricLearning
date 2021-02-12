import sys, os, math
import argparse
import numpy as np
import time
import ConfigParser
import zipfile
import cPickle
import PIL.Image
import PIL.ImageFile
import multiprocessing
import collections
import ctypes
import itertools
import signal
from optparse import OptionParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
plt.figure(figsize=(16,128))
from indexlib_gpu_cpu import *
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
from testbed_utils import *

# Input options
def init_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("--p", dest="PREFIX", help="prefix of caffemodel")
    parser.add_option("--i", dest="iteration", default = None, help="iteration number of model to evaluate. Leave to evaluate for all")
    parser.add_option("--individualLearners", dest="individualLearners", default = None, help="set the number of individual learners to be evaluated")
    parser.add_option("--ignoreQuery", dest="ignoreQuery", action='store_true', default = False, help="ignore query image in the retrieval result list")
    parser.add_option("--saveResult", dest="saveResult", action='store_true', default = False, help="save result images and retreival ids")
    parser.add_option("--selfSearch", dest="selfSearch", action='store_true', default = False, help="query db and retrieval db are same")
    return parser.parse_args()


# Config file
cp = ConfigParser.SafeConfigParser()
cp.add_section('data')
cp.set('data', 'gt_path', 'No')
cp.set('data', 'batch_size', '256')
cp.add_section('feature')
cp.set('feature', 'val_prefix', '')
cp.read('testbed.cfg')
print 'Config values:'
for section in cp.sections():
    print section
    for name, value in cp.items(section):
        print '  %s = %r' % (name, value)

top_n = 7
DISTRACTOR_FOLDER = DISTRACTOR_FILE = QUERY_FOLDER = QUERY_FILE = RETRIEVAL_FOLDER = RETRIEVAL_FILE = GT_PATH = BATCHSIZE = None
MODEL = preprocess = layer_name = val_prefix = ANSWER_FILE = CAFFEMODEL_FOLDER = RESULTS_FOLDER = VAL_FILE = None
gt_dict = None

def evaluate( query, db, N, saveResult, evaluateRecall, ignoreQuery = False):
    index = BruteForceCUBLAS()
    index.fit( db[0] )
    success = np.zeros( N+1 )
    if(saveResult):
    	f  = open(ANSWER_FILE,'w')
    for i in xrange(len(query[1])):
        feat, fname = query[0][i], query[1][i]
        result_idx, result_dists = index.query( feat, N+1 )
        result = [ db[1][r] for r in result_idx ]
	if(ignoreQuery):
		for i,r in enumerate(result):
			if(result[i].split('/')[-1]==fname.split('/')[-1]):
				del result[i]
				break
		#result = result[1:]
        result = result[:N] # index returns more than N
	if(saveResult):
    		canvas1 = np.zeros((256,256,3))
    		canvas1 = buildSquareThumbnail(query[1][i], canvas1)
		f.write(fname.split('/')[-1])
		for n in xrange(N):
			canvas2 = np.zeros((256,256,3))
			canvas2 = buildSquareThumbnail(result[n], canvas2)
			canvas1 = np.hstack((canvas1, canvas2))
			resultName = result[n].split('/')[-1]
			f.write(' ' + str(resultName))
		f.write('\n')
		cv2.imwrite('%s%s%d.jpg'%(RESULTS_IMAGES_FOLDER, query[1][i].split('/')[-1]  + '-', i), canvas1)
	if(evaluateRecall):
		for n in range(1,N+1):
			resultRange = result[:n]
        		success[n] += check_hit(gt_dict, fname, resultRange)
        		#success[n] += average_precision(gt_dict, fname, resultRange)

		# mean across all queries
    success = success / float(len(query[1]))
    if(evaluateRecall):
    	return success[1:], len(query[1])
    else:
	f.close()
	return None, len(query[1])

def evaluate_model(pool, MODEL, WEIGHT, saveResult = False, evaluateRecall = True, ignoreQuery = False, selfSearch = False):
    net, tr = load_net (MODEL , WEIGHT )
    input_size = net.blobs['data'].shape[-1] 
    minibatch=BATCHSIZE

    input_folder = QUERY_FOLDER
    input_size = net.blobs['data'].shape[-1] 
    batch_data, chunk_data = [], []
   
    for item in img_loader_pool(pool, input_folder, tr, preprocess, input_size, minibatch, QUERY_FILE):
	if item is not None:
		batch_data.append(item)
        if len(batch_data) >= minibatch:
            chunk_data.append( extract_batch_feature(net, layer_name, batch_data ) )
            batch_data = []
            sys.stderr.write('.')
            sys.stderr.flush()
    if len(batch_data) > 0:
        chunk_data.append( extract_batch_feature(net, layer_name, batch_data ) )
    mini_feats, mini_keys, _ = zip(*chunk_data)

    features_query = np.vstack(mini_feats)
    keys_query = list( itertools.chain( *mini_keys) )
    query_chunk = features_query, keys_query

    if(not selfSearch):
	    input_folder = RETRIEVAL_FOLDER
	    input_size = net.blobs['data'].shape[-1]
	    batch_data, chunk_data = [], []
	    for item in img_loader_pool(pool, input_folder, tr, preprocess, input_size, minibatch, RETRIEVAL_FILE):
	        if item is not None:
	            batch_data.append(item)
	        if len(batch_data) >= minibatch:
	            chunk_data.append( extract_batch_feature(net, layer_name, batch_data ) )
	            batch_data = []
	            sys.stderr.write('.')
	            sys.stderr.flush()
	    if len(batch_data) > 0:
	        chunk_data.append( extract_batch_feature(net, layer_name, batch_data ) )
	    mini_feats, mini_keys, _ = zip(*chunk_data)
	    features_retrieval = np.vstack(mini_feats)
	    keys_retrieval = list( itertools.chain( *mini_keys) )
    else:
            print 'Using query features for retrieval'
	    features_retrieval = np.copy(features_query)
            keys_retrieval = keys_query

    input_folder = DISTRACTOR_FOLDER
    batch_data, chunk_data = [], []
    if(input_folder!='No'):
	    input_size = net.blobs['data'].shape[-1]
	    for item in img_loader_pool(pool, input_folder, tr, preprocess, input_size, minibatch, DISTRACTOR_FILE):
	        if item is not None:
	            batch_data.append(item)
	        if len(batch_data) >= minibatch:
	            chunk_data.append( extract_batch_feature(net, layer_name, batch_data ) )
	            batch_data = []
	            sys.stderr.write('.')
	            sys.stderr.flush()
	    if len(batch_data) > 0:
	        chunk_data.append( extract_batch_feature(net, layer_name, batch_data ) )
	    mini_feats, mini_keys, _ = zip(*chunk_data)
	    features_retrieval2 = np.vstack(mini_feats)
	    keys_retrieval2 = list( itertools.chain( *mini_keys) )
	    features_retrieval = np.concatenate((features_retrieval2, features_retrieval), axis=0)
	    keys_retrieval = np.concatenate((keys_retrieval2, keys_retrieval), axis=0)
    

    retrieval_chunk = features_retrieval, keys_retrieval

    try:
        del net
        del tr
    	sys.stderr.write('deleted\n')
    	sys.stderr.flush()
    except:
    	sys.stderr.write('couldnt delete\n')
    	sys.stderr.flush()
    	pass   

    recall, n_queries = evaluate( query_chunk,retrieval_chunk, top_n, saveResult, evaluateRecall, ignoreQuery)
    if(recall is not None):
	return recall
    else:
	return None

def get_iter(x):
    return int((x.split('.')[0]).split('_')[-1])

def caffemodel_comp(x, y):
    xind = get_iter(x)
    yind = get_iter(y)
    if xind > yind:
        return 1
    elif xind < yind:
        return -1
    else:
        return 0
    
def get_caffemodel_list(prefix, FOLDER):
    flist = os.listdir(FOLDER)
    caffemodel_list = filter(lambda x: (x.split('.')[-1] == 'caffemodel' and x.startswith(prefix + '_iter')), flist)
    caffemodel_list.sort(caffemodel_comp)
    return caffemodel_list
    

def buildSquareThumbnail(path, canvas):
    try:
        img = cv2.imread(path)
        scale = 256. / max(img.shape)
        if scale >= 1:
            INTERPOLATION = cv2.INTER_LINEAR
        else:
            INTERPOLATION = cv2.INTER_AREA
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=INTERPOLATION)
        canvas[:img.shape[0], :img.shape[1],:] = img
        return canvas
    except:
        print "failed at"
        print path
	return canvas


if __name__ == '__main__':
    (opts, args) = init_options()
    print opts
    sys.stdout.flush()

    # read data config
    DISTRACTOR_FOLDER = cp.get( 'data', 'distractor_folder' )
    DISTRACTOR_FILE = cp.get( 'data', 'distractor_file' )
    QUERY_FOLDER = cp.get( 'data', 'query_folder' )
    QUERY_FILE = cp.get( 'data', 'query_file' )
    RETRIEVAL_FOLDER = cp.get( 'data', 'retrieval_folder' )
    RETRIEVAL_FILE = cp.get( 'data', 'retrieval_file' )
    GT_PATH = cp.get( 'data', 'gt_path' )
    BATCHSIZE = int(cp.get( 'data', 'batch_size' ))

    MODEL = cp.get( 'feature', 'model' )
    preprocess = cp.get( 'feature', 'preprocess' )
    layer_name = cp.get( 'feature', 'layer_name' )
    val_prefix = cp.get( 'feature', 'val_prefix' )
    
    ANSWER_FILE = ''
    CAFFEMODEL_FOLDER = 'snapshots/'
    RESULTS_FOLDER = './results/'
    VAL_FILE = 'validation/val-'
    if(val_prefix is not ''):
    	VAL_FILE = VAL_FILE + val_prefix + '-'

    if(GT_PATH!='No'):
    	gt_dict = dictFromTxt(GT_PATH)
    else:
    	gt_dict = dictFromTrainTxt(QUERY_FILE)

    assert opts.PREFIX is not None
    PREFIX = opts.PREFIX
    pool = multiprocessing.Pool(processes = 6, initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN))
    if(opts.iteration):
        WEIGHT = os.path.join(CAFFEMODEL_FOLDER, PREFIX + '_iter_' + str(opts.iteration) + '.caffemodel')
	if(opts.iteration == '0'):
        	WEIGHT = PREFIX
	if(opts.saveResult):
		if not os.path.exists(RESULTS_FOLDER):
    			os.mkdir(RESULTS_FOLDER)
		RESULTS_IMAGES_FOLDER = RESULTS_FOLDER + PREFIX + '_iter_' + str(opts.iteration) + '/'
		ANSWER_FILE = RESULTS_IMAGES_FOLDER + 'answer.txt'
		if not os.path.exists(RESULTS_IMAGES_FOLDER):
    			os.mkdir(RESULTS_IMAGES_FOLDER)
	if(opts.individualLearners is not None):
		allRecall = []
		print "All Recall --"
		for i in xrange(int(opts.individualLearners)):
			layer_name = 'loss3/l2_norm_' + chr(97+i) +  'branch' 
			#layer_name = 'loss3/classifier_norm_' + ('_f'+str(i+1) if i>0 else '')
			print 'Using layer_name: ' + layer_name
			recall = evaluate_model(pool, MODEL, WEIGHT, opts.saveResult, True, opts.ignoreQuery, opts.selfSearch)
			allRecall.append(recall)	
		for i in xrange(int(opts.individualLearners)):
			print 'Recall(i) ' + str(i) + ': ' + '\t'.join(map(str, allRecall[i]))
	else:        
		recall = evaluate_model(pool, MODEL, WEIGHT, opts.saveResult, True, opts.ignoreQuery, opts.selfSearch)
		print 'Recall: ' + '\t'.join(map(str, recall))
    else:
        VAL_FILE = VAL_FILE + PREFIX + '.recall'
        if not os.path.exists(VAL_FILE):
            flines = ['iter 1 2 3 4 5 6 7 8 9 10\n']
            with open(VAL_FILE,'w') as f:
                f.writelines(flines) 
	    
        while True:
            caffemodel_list = get_caffemodel_list(PREFIX, CAFFEMODEL_FOLDER)
            with open(VAL_FILE,'r') as f:
                    flines = f.readlines()
	    calculated_models = map(lambda x: x.strip().split()[0], flines[1:])   

	    for caffemodel in caffemodel_list:
		caffemodel_iter = str(get_iter(caffemodel))
		if(caffemodel_iter in calculated_models):
			continue
		
                WEIGHT = os.path.join(CAFFEMODEL_FOLDER, caffemodel)
                line = caffemodel_iter
                line += '\n'
                flines += [line]
                try:
                    with open(VAL_FILE,'w') as f:
                        f.writelines(flines) 
                except IOError as e:
                    print "I/O error({0}): {1}".format(e.errno, e.strerror)
                    # Adding failsafe for the case when someone messes up and supercom disk quota exceeds
                    pass

                recall = evaluate_model(pool, MODEL, WEIGHT, False, True, opts.ignoreQuery, opts.selfSearch)
	        sys.stdout.write('recall ')
	        print recall
	        sys.stdout.flush()

                line = caffemodel_iter
                for r in recall:
                    line += ' ' + str(r)
                line += '\n'

                with open(VAL_FILE,'r') as f:
                    flines = f.readlines()
	        calculated_models = map(lambda x: x.strip().split()[0], flines[1:])   
		flines[calculated_models.index(caffemodel_iter)+1] = line
                try:
                    with open(VAL_FILE,'w') as f:
                        f.writelines(flines) 
                except IOError:
                    pass

		break
	    time.sleep(10)
	    continue
 
    
