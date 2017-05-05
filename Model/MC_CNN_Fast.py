import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d, relu
from tqdm import tqdm

from Loader import *

mPath  = os.path.dirname(os.path.abspath("MC_CNN_Fast.py")) + "/Model/"
wPath = os.path.dirname(os.path.abspath("MC_CNN_Fast.py")) + "/Weights/"

start_epoch = 0
n_epochs = 100

batch_size = 64
f_size = 3
img_size = 9

N_IMGS = 400000
N_TRAIN = 320000
AllPatches = loadAllImages(N_IMGS)

RefPatches = AllPatches[0]
PosPatches = AllPatches[1]
NegPatches = AllPatches[2]
print("Data Loaded")

X_l = T.ftensor4()
X_r_plus = T.ftensor4()
X_r_minus = T.ftensor4()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    fan_in = np.prod(shape[1:])
    fan_out = (shape[0] * np.prod(shape[2:]))

    local=np.random.randn(*shape)
    W_bound = np.sqrt(2.0/(fan_in))

    return theano.shared(floatX(local*W_bound))

def init_bias(shape):
    b_values = np.zeros((shape[0],), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, borrow=True)
    return b

def Adam_updates(loss, all_params, learning_rate=0.0002, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates

def l2_norm_layer(ip):
    norm = T.inv(T.sqrt(((ip**2).sum(axis=(1,2,3)))))
    sq = T.reshape(norm, (batch_size,1,1,1))
    op = ip*sq

    return op

'''layer size specs'''
model_size = []

model_size.append((64,1,f_size,f_size))
model_size.append((64,64,f_size,f_size))
model_size.append((64,64,3,3))
model_size.append((64,64,3,3))

def model(X, w1, w2, w3, w4):

    l1 = relu((conv2d(X,w1, border_mode='full')))
    l2 = relu((conv2d(l1,w2, border_mode='valid')))
    l3 = relu((conv2d(l2,w3,border_mode='full')))
    l4 = conv2d(l3,w4,border_mode='valid')

    output = l2_norm_layer(l4)

    return output

w1 = init_weights(model_size[0])
w2 = init_weights(model_size[1])
w3 = init_weights(model_size[2])
w4 = init_weights(model_size[3])

params = [w1, w2, w3, w4]

nn_output_left = model(X_l, *params)
nn_output_right_pos = model(X_r_plus, *params)
nn_output_right_neg = model(X_r_minus, *params)

s_plus=((nn_output_left*nn_output_right_pos).sum(axis=(1,2,3)))
s_minus=((nn_output_left*nn_output_right_neg).sum(axis=(1,2,3)))

margin = (0.2 + s_minus - s_plus)
margin = T.switch(margin>0, margin,0)
loss_final = margin.sum()

print("Compiling")
updater = Adam_updates(loss_final, params)
train_model = theano.function(inputs=[X_l,X_r_plus,X_r_minus], outputs = [margin,loss_final], updates= updater)
test_model = theano.function(inputs=[X_l,X_r_plus,X_r_minus], outputs = [margin,loss_final])
print("Compiled Model")

f_train=open(mPath + "Epoch_stats_training.txt","w",0)
f_train.write("Epoch:	    Positives	  Negatives          Error        Accuracy            Loss Cost \n")

f_test=open(mPath + "Epoch_stats_testing.txt","w",0)
f_test.write("Epoch:	    Positives	  Negatives          Error        Accuracy            Loss Cost \n")

if(start_epoch > 0):
    print("Loading Weights OF epoch %d"%(start_epoch-1))
    layer_idx=0
    for layer in tqdm(range(len(params))):
        current_layer = np.load(wPath+"weights_epoch_%d_layer_%d.npy"%(start_epoch-1,layer_idx))
        params[layer_idx].set_value(current_layer)
        layer_idx+=1

for epoch in range(start_epoch,n_epochs):

    RefPatches = np.float32(RefPatches)
    PosPatches = np.float32(PosPatches)
    NegPatches = np.float32(NegPatches)

    counter_error = 0;counter_s_plus = 0;counter_s_minus = 0;counter_cost = 0;listicle = []
    print("Training")
    print("Epoch: %d"%epoch)
    for idx in tqdm(range(0,N_TRAIN, batch_size)):
        ref_batch = RefPatches[idx:idx+batch_size,:,:,:]
        pos_batch = PosPatches[idx:idx+batch_size,:,:,:]
        neg_batch = NegPatches[idx:idx+batch_size,:,:,:]

        classify = train_model(ref_batch, pos_batch, neg_batch)
        counter_error+=np.count_nonzero(classify[0])
    	counter_s_minus+=np.count_nonzero(classify[0])
    	counter_s_plus+=(batch_size-np.count_nonzero(classify[0]))
        counter_cost+=classify[1]

    listicle=listicle+[epoch,counter_s_plus,counter_s_minus,counter_error,(float(counter_s_plus*100.)/(counter_s_plus + counter_s_minus)),counter_cost]
    for i in listicle:
        f_train.write(str(i)+"             ")
    f_train.write('\n')
    print ("Train Accuracy:")
    print(listicle)

    counter_error = 0;counter_s_plus = 0;counter_s_minus = 0;counter_cost = 0;listicle = []
    print("Testing on seperate batch")
    for idx in range(N_TRAIN,N_IMGS, batch_size):
        ref_batch = RefPatches[idx:idx+batch_size,:,:,:]
        pos_batch = PosPatches[idx:idx+batch_size,:,:,:]
        neg_batch = NegPatches[idx:idx+batch_size,:,:,:]

        classify = test_model(ref_batch, pos_batch, neg_batch)
        print("Processing BATCH: "+str(idx)+" Hinge Loss sum: "+str(classify[1]))
        counter_error+=np.count_nonzero(classify[0])
    	counter_s_minus+=np.count_nonzero(classify[0])
    	counter_s_plus+=(batch_size-np.count_nonzero(classify[0]))
        counter_cost+=classify[1]

    listicle=listicle+[epoch,counter_s_plus,counter_s_minus,counter_error,(float(counter_s_plus*100.)/(counter_s_plus + counter_s_minus)),counter_cost]
    for i in listicle:
        f_test.write(str(i)+"             ")
    f_test.write('\n')
    print ("Test Accuracy:")
    print(listicle)

    print("Saving weights")
    layer_idx=0
    for layer in tqdm(range(len(params))):
        current_layer = params[layer].get_value()
        np.save(wPath + "weights_epoch_%d_layer_%d.npy"%(epoch,layer_idx), current_layer)
        layer_idx+=1

f_train.close()
f_test.close()
