#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from conv_blocks import Conv2dBNReLUBlock, DenseBlockNoOutCat, TCNDepthWise, DenseBlockNoOutCatFM

class CSeqUNetDense(nn.Module):

    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(self, input_dim, nlayer, n_units, target_dim, use_seqmodel, masking_or_mapping,
            output_activation, rmax,
            approx_method, loss_function,
            use_batchnorm, use_convbnrelu, use_act,
            memory_efficient, n_outputs=2):
        super(CSeqUNetDense, self).__init__()
        """
        In the CHiME-4 paper, the arguments are:
        input_dim=x*257
        nlayer=2
        n_units=512
        target_dim=257
        use_seqmodel=1
        masking_or_mapping=1
        output_activation='linear'
        rmax=5
        approx_method='MSA-RIx2'
        loss_function='l1loss'
        use_batchnorm=3
        use_convbnrelu=2
        use_act=2
        memory_efficient=1
        """

        approx_method = approx_method.split('-')
        self.approx_method = approx_method

        if loss_function not in ['l1loss', 'l2loss']:raise
        self.loss_function = loss_function

        if target_dim not in [257]: raise

        if input_dim % target_dim != 0: raise
        in_channels = input_dim // target_dim
        assert in_channels >= 1

        t_ksize = 3
        #
        #              257,   1                                   2/4
        #(257-5)/2+1 = 127,  32 ---5*32--> 32  +  32  ---5*32---> 32
        #(127-3)/2+1 =  63,  32 ---5*32--> 32  +  32  ---5*32---> 32
        #(63-3)/2+1  =  31,  32 ---5*32--> 32  +  32  ---5*32---> 32
        #(31-3)/2+1  =  15,  64 ---5*64--> 64  +  64  ---5*64---> 64
        #(15-3)/2+1  =   7, 128                +  128
        #(7-3)/2+1   =   3, 256                +  256
        #(3-3)/1+1   =   1, 512                +  512
        # 
        self.conv0 = nn.Conv2d(in_channels,32,(t_ksize,5),stride=(1,2),padding=(t_ksize//2,0))
        self.eden0 = DenseBlockNoOutCatFM(32,32,(t_ksize,3),127,n_layers=5,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.conv1 = Conv2dBNReLUBlock(32,32,(t_ksize,3),stride=(1,2),padding=(t_ksize//2,0),use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.eden1 = DenseBlockNoOutCatFM(32,32,(t_ksize,3),63,n_layers=5,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.conv2 = Conv2dBNReLUBlock(32,32,(t_ksize,3),stride=(1,2),padding=(t_ksize//2,0),use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.eden2 = DenseBlockNoOutCatFM(32,32,(t_ksize,3),31,n_layers=5,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.conv3 = Conv2dBNReLUBlock(32,64,(t_ksize,3),stride=(1,2),padding=(t_ksize//2,0),use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.eden3 = DenseBlockNoOutCatFM(64,64,(t_ksize,3),15,n_layers=5,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.conv4 = Conv2dBNReLUBlock(64,128,(t_ksize,3),stride=(1,2),padding=(t_ksize//2,0),use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.conv5 = Conv2dBNReLUBlock(128,256,(t_ksize,3),stride=(1,2),padding=(t_ksize//2,0),use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.conv6 = Conv2dBNReLUBlock(256,512,(t_ksize,3),stride=(1,1),padding=(t_ksize//2,0),use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        encoder_dim = 512
        input_dim   = encoder_dim

        if use_seqmodel == 0:
            for i in range(nlayer):
                input_dim = input_dim if i == 0 else n_units*2
                self.add_module("bilstm-%d"%i, nn.LSTM(input_dim, n_units, 1, batch_first=True, dropout=0.0, bidirectional=True))
                # Setting forget gate bias to a 2.0
                self["bilstm-%d"%i].bias_hh_l0.data[n_units:2*n_units] = 1.0
                self["bilstm-%d"%i].bias_ih_l0.data[n_units:2*n_units] = 1.0
                self["bilstm-%d"%i].bias_hh_l0_reverse.data[n_units:2*n_units] = 1.0
                self["bilstm-%d"%i].bias_ih_l0_reverse.data[n_units:2*n_units] = 1.0
        else:
            tcn_classname = TCNDepthWise
            for ii in range(1,nlayer+1):
                self.add_module('tcn-conv%d-0'%ii, tcn_classname(input_dim,input_dim,t_ksize,use_batchnorm=use_batchnorm,use_act=use_act,dilation=1))
                self.add_module('tcn-conv%d-1'%ii, tcn_classname(input_dim,input_dim,t_ksize,use_batchnorm=use_batchnorm,use_act=use_act,dilation=2))
                self.add_module('tcn-conv%d-2'%ii, tcn_classname(input_dim,input_dim,t_ksize,use_batchnorm=use_batchnorm,use_act=use_act,dilation=4))
                self.add_module('tcn-conv%d-3'%ii, tcn_classname(input_dim,input_dim,t_ksize,use_batchnorm=use_batchnorm,use_act=use_act,dilation=8))
                self.add_module('tcn-conv%d-4'%ii, tcn_classname(input_dim,input_dim,t_ksize,use_batchnorm=use_batchnorm,use_act=use_act,dilation=16))
                self.add_module('tcn-conv%d-5'%ii, tcn_classname(input_dim,input_dim,t_ksize,use_batchnorm=use_batchnorm,use_act=use_act,dilation=32)) 

        self.output_activation = output_activation
        self.rmax = rmax
        if masking_or_mapping == 0:
            #masking
            initial_bias = 0.0
        else:
            #mapping
            initial_bias = 0.0

        #
        #              257,   1                                   2/4
        #(257-5)/2+1 = 127,  32 ---5*32--> 32  +  32  ---5*32---> 32
        #(127-3)/2+1 =  63,  32 ---5*32--> 32  +  32  ---5*32---> 32
        #(63-3)/2+1  =  31,  32 ---5*32--> 32  +  32  ---5*32---> 32
        #(31-3)/2+1  =  15,  64 ---5*64--> 64  +  64  ---5*64---> 64
        #(15-3)/2+1  =   7, 128                +  128
        #(7-3)/2+1   =   3, 256                +  256
        #(3-3)/1+1   =   1, 512                +  512
        #
        self.deconv0 = Conv2dBNReLUBlock(encoder_dim+input_dim,256,(t_ksize,3),stride=(1,1),padding=(t_ksize//2,0),use_deconv=1,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.deconv1 = Conv2dBNReLUBlock(2*256,128,(t_ksize,3),stride=(1,2),padding=(t_ksize//2,0),use_deconv=1,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.deconv2 = Conv2dBNReLUBlock(2*128,64,(t_ksize,3),stride=(1,2),padding=(t_ksize//2,0),use_deconv=1,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.dden2 = DenseBlockNoOutCatFM(64+64,64,(t_ksize,3),15,n_layers=5,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.deconv3 = Conv2dBNReLUBlock(64,32,(t_ksize,3),stride=(1,2),padding=(t_ksize//2,0),use_deconv=1,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.dden3 = DenseBlockNoOutCatFM(32+32,32,(t_ksize,3),31,n_layers=5,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.deconv4 = Conv2dBNReLUBlock(32,32,(t_ksize,3),stride=(1,2),padding=(t_ksize//2,0),use_deconv=1,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.dden4 = DenseBlockNoOutCatFM(32+32,32,(t_ksize,3),63,n_layers=5,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.deconv5 = Conv2dBNReLUBlock(32,32,(t_ksize,3),stride=(1,2),padding=(t_ksize//2,0),use_deconv=1,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.dden5 = DenseBlockNoOutCatFM(32+32,32,(t_ksize,3),127,n_layers=5,use_batchnorm=use_batchnorm,use_act=use_act,use_convbnrelu=use_convbnrelu,memory_efficient=memory_efficient)
        self.deconv6 = nn.ConvTranspose2d(32,n_outputs,(t_ksize,5),stride=(1,2),padding=(t_ksize//2,0))
        self.deconv6.bias.data[:] = initial_bias

        self.target_dim     = target_dim
        self.nlayer         = nlayer
        self.n_units        = n_units
        self.use_seqmodel   = use_seqmodel
        self.masking_or_mapping = masking_or_mapping
        self.n_outputs      = n_outputs

    def forward(self, x_ins, device, input_dropout_rate=0.0, hidden_dropout_rate=0.0, dilation_dropout_rate=0.0):
        """
        x_ins: list of 2D tensors
        """
        batchsize   = len(x_ins)
        ilenvec     = [x_in.shape[0] for x_in in x_ins]
        N           = max(ilenvec)

        batch = np.concatenate(x_ins,axis=0)
        batch = torch.from_numpy(batch)
        batch = torch.split(batch,ilenvec,dim=0)
        batch = pad_sequence(batch,batch_first=True,padding_value=0) #[batchsize, N, -1]

        batch = batch.to(device)

        if input_dropout_rate > 0.0:
            batch = F.dropout(batch,p=input_dropout_rate,training=True,inplace=False)

        batch = batch.view([batchsize,N,-1,self.target_dim]) #[batchsize, N, -1, n_freqs]
        batch = batch.transpose(1,2) #[batchsize, -1, N, n_freqs]

        all_conv_batch = []
        for cc in range(10):
            conv_link_name = 'conv%d'%cc
            if hasattr(self, conv_link_name):
                batch = self[conv_link_name](batch)
                eden_link_name = 'eden%d'%cc
                if hasattr(self, eden_link_name):
                    batch = self[eden_link_name](batch)
                all_conv_batch.append(batch)
            else:
                break
        #batch.shape is [batchsize, self.n_units, N, 1]

        if self.use_seqmodel == 0:
            batch = batch.squeeze(dim=-1) #[batchsize, self.n_units, N]
            batch = batch.transpose(1,2) #[batchsize, N, self.n_units]
            batch = self.propagate_full_sequence(batch, dropout_rate=hidden_dropout_rate) #[batchsize, N, 2*self.n_units]
            batch = batch.transpose(1,2) #[batchsize, 2*self.n_units, N]
            batch = batch.unsqueeze(dim=-1) #[batchsize, 2*self.n_units, N, 1]
        else:
            batch = batch.view([batchsize,self.n_units,N]) #[batchsize, self.n_units, N]
            for ii in range(1, self.nlayer+1):
                for cc in range(20):
                    conv_link_name = 'tcn-conv%d-%d'%(ii,cc)
                    if hasattr(self, conv_link_name):
                        batch = self[conv_link_name](batch,
                                hidden_dropout_rate=hidden_dropout_rate,
                                dilation_dropout_rate=dilation_dropout_rate)
                    else:
                        break
            batch = batch.unsqueeze(dim=-1) #[batchsize, self.n_units, N, 1]

        for cc in range(10):
            deconv_link_name = 'deconv%d'%cc
            if hasattr(self, deconv_link_name):
                if cc-1 >= 0 and hasattr(self, 'dden%d'%(cc-1)):
                    batch = self[deconv_link_name](batch)
                else:
                    batch = self[deconv_link_name](torch.cat([batch,all_conv_batch[-1-cc]],dim=1))
                dden_link_name = 'dden%d'%cc
                if hasattr(self, dden_link_name):
                    batch = self[dden_link_name](torch.cat([batch,all_conv_batch[-1-cc-1]],dim=1))
            else:
                break
        #batch.shape is [batchsize, -1, N, self.target_dim]

        batch = batch.transpose(1,2) #[batchsize, N, -1, self.target_dim]
        batch = batch.reshape([batchsize,N,-1]) #[batchsize, N, num_speakers*n_outputs*self.target_dim]
        batch = [batch[bb,:utt_len] for bb,utt_len in enumerate(ilenvec)]
        batch = torch.cat(batch,dim=0) #[-1,n_outputs*target_dim]

        if self.masking_or_mapping == 0:
            #masking
            if self.pitactivation == 'linear':
                activations = torch.clamp(batch,-self.rmax,self.rmax)
            else:
                raise
        else:
            #mapping
            if self.output_activation == 'linear':
                activations = batch
            else:
                raise
        self.activations = activations #[n_frames,n_outputs*target_dim]

    def get_loss(self, ins, device):

        if self.loss_function.startswith('l2'):
            raise
        else:
            loss_type = torch.abs

        y_reals = np.concatenate(ins[0][1], axis=0)
        y_imags = np.concatenate(ins[0][2], axis=0)
        y_reals = torch.from_numpy(y_reals).to(device)
        y_imags = torch.from_numpy(y_imags).to(device)

        activations_reals, activations_imags = torch.chunk(self.activations,self.n_outputs,dim=-1)

        if self.masking_or_mapping == 0:
            x_reals = np.concatenate(ins[1][1], axis=0)
            x_imags = np.concatenate(ins[1][2], axis=0)
            x_reals = torch.from_numpy(x_reals).to(device)
            x_imags = torch.from_numpy(x_imags).to(device)
            #(a+b*i)*(c+d*i) = ac-bd + (ad+bc)*i
            activations_reals, activations_imags = x_reals*activations_reals-x_imags*activations_imags, x_reals*activations_imags+x_imags*activations_reals
        else:
            pass

        est_y_reals, est_y_imags = activations_reals, activations_imags

        ret = [torch.tensor(0.0,device=device)]

        if 'MSA' in self.approx_method:
            y_mags = torch.sqrt(y_reals**2+y_imags**2+1e-5)
            est_y_mags = torch.sqrt(est_y_reals**2+est_y_imags**2+1e-5)
            loss_mags = torch.mean(loss_type(est_y_mags - y_mags))
            ret[0] += loss_mags
            ret.append(loss_mags)

        if 'RIx2' in self.approx_method:
            loss_reals = torch.mean(loss_type(y_reals - est_y_reals))
            loss_imags = torch.mean(loss_type(y_imags - est_y_imags))
            ret[0] += ((loss_reals+loss_imags))
            ret.append(loss_reals)
            ret.append(loss_imags)

        return ret

    def propagate_one_layer(self, batch, layer, dropout_rate=0.0):
        batch, (_, _) = self['bilstm-%d'%layer](batch)
        return F.dropout(batch, p=dropout_rate, training=True, inplace=False) if dropout_rate > 0.0 else batch
 
    def propagate_full_sequence(self, batch, dropout_rate=0.0):
        for ll in range(self.nlayer-1):
            batch = self.propagate_one_layer(batch, ll, dropout_rate=dropout_rate)
        batch = self.propagate_one_layer(batch, self.nlayer-1, dropout_rate=0.0)
        return batch
