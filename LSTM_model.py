import numpy as np

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))


class Network():
    def __init__(self, syn_arch : list, neu_arch : list, n_ext_in:int,n_ext_out:int ):
        super(Network, self).__init__()

    
        self.syn_arch = syn_arch
        self.neu_arch = neu_arch

        self.synapses = []
        self.neurons = []

        self.n_ext_in = n_ext_in
        self.n_ext_out = n_ext_out

    def forward(self, inp):

        inp = np.repeat(inp, self.syn_arch[0]).reshape(self.n_ext_in, 1, self.syn_arch[0], 1)

        X = self.synapses[0].forward(inp)
        post = self.neurons[0].forward(X)
        self.synapses[0].update(inp,post)

        pre = post
        for tick in range(2):
            X = self.synapses[1].forward(pre)
            post = self.neurons[0].forward(X)
            self.synapses[1].update(pre, post)
            pre = post


        X = self.synapses[2].forward(pre)
        self.synapses[2].update(pre, X)

        X = X[:,:,0,:]
        return X


class Neurons():

    def __init__(self, n_neu, arch): #in_channels, hid_size, out_channels):
        super(Neurons, self).__init__()

        self.arch = arch
        in_channels = arch[0]
        hid_size = arch[1]
        out_channels = arch[2]
        
        self.n_neu = n_neu

        self.in_channels = in_channels
        self.hid_size = hid_size
        self.out_channels = out_channels

        self.hidden_state = np.random.normal(0,0.1, (n_neu,1,hid_size, 1))
        self.cell_state = np.random.normal(0,0.1, (n_neu,1,hid_size, 1))

        ## Weights:
        self.Wf_init = np.random.normal(0,0.1, (1,1,in_channels + hid_size, hid_size))
        self.Wi_init = np.random.normal(0,0.1, (1,1,in_channels + hid_size, hid_size))
        self.Wc_init = np.random.normal(0,0.1, (1,1,in_channels + hid_size, hid_size))
        self.Wo_init = np.random.normal(0,0.1, (1,1,in_channels + hid_size, hid_size))

        self.Wout_init = np.random.normal(0,0.1, (1,1,hid_size, out_channels))

        self.Bf_init = np.zeros((1,1,hid_size, 1))
        self.Bi_init = np.zeros((1,1,hid_size, 1))
        self.Bc_init = np.zeros((1,1,hid_size, 1))
        self.Bo_init = np.zeros((1,1,hid_size, 1))
        #self.Bout = np.zeros((1,1,out_channels, 1))

    def forward(self, inp):

        #print(inp.shape, self.hidden_state.shape)
        x = np.concatenate((inp, self.hidden_state), axis=2)

        f = sigmoid( np.einsum('lmbn,lmbc->lmnc', self.Wf, x)+self.Bf)
        i = sigmoid( np.einsum('lmbn,lmbc->lmnc', self.Wi, x)+self.Bi)

        c = np.tanh( np.einsum('lmbn,lmbc->lmnc', self.Wc, x)+self.Bc)

        self.cell_state = f * self.cell_state + i * c

        o = sigmoid( np.einsum('lmbn,lmbc->lmnc', self.Wo, x)+self.Bo)


        self.hidden_state = o * np.tanh(self.cell_state)


        out = np.tanh( np.einsum('lmbn,lmbc->lmnc', self.Wout, self.hidden_state))

        return out


    def shape_params(self, flat_params):

        n_in = self.in_channels
        n_hid = self.hid_size
        n_out = self.out_channels

        m=0

        ##weights
        Wf = flat_params[m:m+ (n_in+n_hid)*n_hid ].reshape(self.Wf_init.shape)
        m+= (n_in+n_hid)*n_hid
        Wi = flat_params[m:m+ (n_in+n_hid)*n_hid ].reshape(self.Wi_init.shape)
        m+= (n_in+n_hid)*n_hid
        Wc = flat_params[m:m+ (n_in+n_hid)*n_hid ].reshape(self.Wc_init.shape)
        m+= (n_in+n_hid)*n_hid
        Wo = flat_params[m:m+ (n_in+n_hid)*n_hid ].reshape(self.Wo_init.shape)
        m+= (n_in+n_hid)*n_hid

        Wout = flat_params[m:m+ (n_hid)*n_out ].reshape(self.Wout_init.shape)
        m+= n_hid*n_out

        ##biases
        Bf = flat_params[m:m+n_hid].reshape(self.Bf_init.shape)
        m+= n_hid
        Bi = flat_params[m:m+n_hid].reshape(self.Bi_init.shape)
        m+= n_hid
        Bc = flat_params[m:m+n_hid].reshape(self.Bc_init.shape)
        m+= n_hid
        Bo = flat_params[m:m+n_hid].reshape(self.Bo_init.shape)
        m+= n_hid
        
        return (Wf, Wi, Wc, Wo, Wout, Bf, Bi, Bc, Bo)



    def set_params(self, flat_params, flat_params2, indx2):
        
        Wf_i, Wi_i, Wc_i, Wo_i, Wout_i, Bf_i, Bi_i, Bc_i, Bo_i= self.shape_params(flat_params)
        
        n_i, n_h, n_o = self.arch
        n_neu = self.n_neu

        self.Wf = np.ones(( n_neu, 1, n_i+n_h, n_h)) * Wf_i
        self.Wi = np.ones(( n_neu, 1, n_i+n_h, n_h)) * Wi_i
        self.Wc = np.ones(( n_neu, 1, n_i+n_h, n_h)) * Wc_i
        self.Wo = np.ones(( n_neu, 1, n_i+n_h, n_h)) * Wo_i
        self.Wout = np.ones(( n_neu, 1, n_h, n_o))   * Wout_i

        self.Bf = np.ones(( n_neu, 1,  n_h,1)) * Bf_i
        self.Bi = np.ones(( n_neu, 1,  n_h,1)) * Bi_i
        self.Bc = np.ones(( n_neu, 1,  n_h,1)) * Bc_i
        self.Bo = np.ones(( n_neu, 1,  n_h,1)) * Bo_i


        Wf_2, Wi_2, Wc_2, Wo_2, Wout_2, Bf_2, Bi_2, Bc_2, Bo_2 = self.shape_params(flat_params2)

        #'''
        ##Set parameters of a different neural unit in random places
        self.Bf[indx2,:,:,:] *= Bf_2
        self.Bi[indx2,:,:,:] *= Bi_2
        self.Bc[indx2,:,:,:] *= Bc_2
        self.Bo[indx2,:,:,:] *= Bo_2

        self.Wf[indx2,:,:,:] *= Wf_2
        self.Wi[indx2,:,:,:] *= Wi_2
        self.Wc[indx2,:,:,:] *= Wc_2
        self.Wo[indx2,:,:,:] *= Wo_2
        self.Wout[indx2,:,:,:] *= Wout_2
        #'''


    def get_params(self):

        flat_params = np.concatenate(
                [
                    self.Wf_init.flatten(),
                    self.Wi_init.flatten(),
                    self.Wc_init.flatten(),
                    self.Wo_init.flatten(),
                    self.Wout_init.flatten(),

                    self.Bf_init.flatten(),
                    self.Bi_init.flatten(),
                    self.Bc_init.flatten(),
                    self.Bo_init.flatten(),
                    #self.Bout.flatten(),
                    ]
                )

        return flat_params




class Synapses():

    def __init__(self, n_pre, n_post, arch): #in_channels, hid_size, out_channels):
        super(Synapses, self).__init__()

        self.n_post = n_post
        self.n_pre = n_pre
    
        self.arch = arch
        n_in = arch[0]
        hid_size = arch[1]
        n_out = arch[2]

        self.n_in = n_in 
        n_in2 = n_in*2
        self.n_in2 = n_in2
        self.hid_size = hid_size
        self.n_out = n_out
    
        self.hidden_state = np.random.normal(0, 0.1, (n_pre,n_post,hid_size, 1))
        self.cell_state = np.random.normal(0,0.1, (n_pre,n_post,hid_size, 1))

        self.broad = np.ones((n_pre, n_post, n_in, 1))
        self.broad2 = np.ones((n_post, n_pre, n_in, 1))

        #self.broad_update = np.ones((n_post, n_pre , n_in, 1))

        ## Weights:
        self.Wf_init = np.random.normal(0,0.1, (1,1,n_in2 + hid_size, hid_size))
        self.Wi_init = np.random.normal(0,0.1, (1,1,n_in2 + hid_size, hid_size))
        self.Wc_init = np.random.normal(0,0.1, (1,1,n_in2 + hid_size, hid_size))
        self.Wo_init = np.random.normal(0,0.1, (1,1,n_in2 + hid_size, hid_size))


        self.Bf_init = np.zeros((1,1,hid_size, 1)) 
        self.Bi_init = np.zeros((1,1,hid_size, 1)) 
        self.Bc_init = np.zeros((1,1,hid_size, 1)) 
        self.Bo_init = np.zeros((1,1,hid_size, 1)) 
        #self.Bout = np.zeros((1,1,n_out, 1)) 


    def forward(self, inp):

        broad_inp =np.einsum('abcd,aocd->abcd', self.broad, inp)
        out = self.hidden_state * broad_inp
        out = np.sum(out, 0)

        return out.reshape(self.n_post, 1, self.n_out, 1) / self.n_pre



    def update(self, pre, post):


        broad_pre =np.einsum('abcd,aocd->abcd', self.broad, pre)
        broad_post = np.einsum('abcd,aocd->bacd', self.broad2, post)
        
        x = np.concatenate((broad_pre, broad_post, self.hidden_state), axis=2)

        f = sigmoid( np.einsum('lmbn,lmbc->lmnc', self.Wf, x)+self.Bf)   
        i = sigmoid( np.einsum('lmbn,lmbc->lmnc', self.Wi, x)+self.Bi)   

        c = np.tanh( np.einsum('lmbn,lmbc->lmnc', self.Wc, x)+self.Bc)   

        self.cell_state = f * self.cell_state + i * c

        o = sigmoid( np.einsum('lmbn,lmbc->lmnc', self.Wo, x)+self.Bo)   

        self.hidden_state = o * np.tanh(self.cell_state)

    
    def shape_params(self, flat_params):
        
        n_in = self.n_in2
        n_hid = self.hid_size
        n_out = self.n_out

        n_pre = self.n_pre
        n_post = self.n_post

        m=0

        ##weights
        Wf = flat_params[m:m+ (n_in+n_hid)*n_hid ].reshape(1, 1, n_in+n_hid, n_hid  )
        m+= (n_in+n_hid)*n_hid
        Wi = flat_params[m:m+ (n_in+n_hid)*n_hid ].reshape(1, 1, n_in+n_hid, n_hid )
        m+= (n_in+n_hid)*n_hid
        Wc = flat_params[m:m+ (n_in+n_hid)*n_hid ].reshape(1, 1, n_in+n_hid, n_hid )
        m+= (n_in+n_hid)*n_hid
        Wo = flat_params[m:m+ (n_in+n_hid)*n_hid ].reshape(1, 1, n_in+n_hid, n_hid )
        m+= (n_in+n_hid)*n_hid
 
        ##biases
        Bf = flat_params[m:m+n_hid].reshape(self.Bf_init.shape)
        m+= n_hid
        Bi = flat_params[m:m+n_hid].reshape(self.Bi_init.shape)
        m+= n_hid
        Bc = flat_params[m:m+n_hid].reshape(self.Bc_init.shape)
        m+= n_hid
        Bo = flat_params[m:m+n_hid].reshape(self.Bo_init.shape)
        m+= n_hid
        
    
        return (Wf, Wi, Wc, Wo, Bf, Bi, Bc, Bo )


    

    def set_params(self, flat_params, flat_params2, indx2, drop_portion=0.5, multiple=False):

        Wf_i, Wi_i, Wc_i, Wo_i,  Bf_i, Bi_i, Bc_i, Bo_i = self.shape_params(flat_params)
        
        _ , n_h, n_o = self.arch
        n_i = self.n_in2

        n_pre = self.n_pre
        n_post = self.n_post
        
        self.Wf = np.ones(( n_pre, n_post, n_i+n_h, n_h)) * Wf_i
        self.Wi = np.ones(( n_pre, n_post, n_i+n_h, n_h)) * Wi_i
        self.Wc = np.ones(( n_pre, n_post, n_i+n_h, n_h)) * Wc_i
        self.Wo = np.ones(( n_pre, n_post, n_i+n_h, n_h)) * Wo_i

        self.Bf = np.ones(( n_pre, n_post,  n_h,1)) * Bf_i
        self.Bi = np.ones(( n_pre, n_post,  n_h,1)) * Bi_i
        self.Bc = np.ones(( n_pre, n_post,  n_h,1)) * Bc_i
        self.Bo = np.ones(( n_pre, n_post,  n_h,1)) * Bo_i
        
        if multiple:
            #'''
            ##Set a parameters for other synaptic units
            ##
            Wf_2, Wi_2, Wc_2, Wo_2,  Bf_2, Bi_2, Bc_2, Bo_2 = self.shape_params(flat_params2)

            self.Bf[indx2, :,:, :] *= Bf_2
            self.Bi[indx2, :,:, :] *= Bi_2
            self.Bc[indx2, :,:, :] *= Bc_2
            self.Bo[indx2, :,:, :] *= Bo_2
            #Bout = Bout.at[:,indx2, :,:, :].set(Bout_2)

            self.Wf[indx2, :,:, :] *= Wf_2
            self.Wi[indx2, :,:,:] *= Wi_2
            self.Wc[indx2, :,:,:] *= Wc_2
            self.Wo[indx2, :,:,:] *= Wo_2
            #'''

        #'''
        ##dropout: set random synapses to zero
        drop_mat = np.random.choice([0,1], (n_pre, n_post,1,1), p=[drop_portion,1-drop_portion])

        self.Bf *= drop_mat
        self.Bi *= drop_mat 
        self.Bc *= drop_mat 
        self.Bo *= drop_mat 
        #Bout = Bout *drop_mat 

        self.Wf *= drop_mat
        self.Wi *= drop_mat
        self.Wc *= drop_mat
        self.Wo *= drop_mat
        #'''


    def get_params(self):

        flat_params = np.concatenate(
                [
                    self.Wf_init.flatten(),
                    self.Wi_init.flatten(),
                    self.Wc_init.flatten(),
                    self.Wo_init.flatten(),

                    self.Bf_init.flatten(),
                    self.Bi_init.flatten(),
                    self.Bc_init.flatten(),
                    self.Bo_init.flatten(),
                    ]
                )

        return flat_params

