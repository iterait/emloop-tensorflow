Search.setIndex({docnames:["cxflow_tensorflow/cxflow_tensorflow","cxflow_tensorflow/cxflow_tensorflow.hooks","cxflow_tensorflow/cxflow_tensorflow.metrics","cxflow_tensorflow/cxflow_tensorflow.models","cxflow_tensorflow/cxflow_tensorflow.models.blocks","cxflow_tensorflow/cxflow_tensorflow.models.conv","cxflow_tensorflow/cxflow_tensorflow.models.conv_blocks","cxflow_tensorflow/cxflow_tensorflow.ops","cxflow_tensorflow/cxflow_tensorflow.utils","cxflow_tensorflow/index","getting_started","index","multi_gpu","regularization","tutorial"],envversion:51,filenames:["cxflow_tensorflow/cxflow_tensorflow.rst","cxflow_tensorflow/cxflow_tensorflow.hooks.rst","cxflow_tensorflow/cxflow_tensorflow.metrics.rst","cxflow_tensorflow/cxflow_tensorflow.models.rst","cxflow_tensorflow/cxflow_tensorflow.models.blocks.rst","cxflow_tensorflow/cxflow_tensorflow.models.conv.rst","cxflow_tensorflow/cxflow_tensorflow.models.conv_blocks.rst","cxflow_tensorflow/cxflow_tensorflow.ops.rst","cxflow_tensorflow/cxflow_tensorflow.utils.rst","cxflow_tensorflow/index.rst","getting_started.rst","index.rst","multi_gpu.rst","regularization.rst","tutorial.rst"],objects:{"":{cxflow_tensorflow:[0,0,0,"-"]},"cxflow_tensorflow.BaseModel":{SIGNAL_MEAN_NAME:[0,2,1,""],SIGNAL_VAR_NAME:[0,2,1,""],TRAINING_FLAG_NAME:[0,2,1,""],TRAIN_OP_NAME:[0,2,1,""],__init__:[0,3,1,""],_create_model:[0,3,1,""],_create_session:[0,3,1,""],_create_train_ops:[0,3,1,""],_initialize_variables:[0,3,1,""],_restore_checkpoint:[0,3,1,""],_restore_model:[0,3,1,""],graph:[0,2,1,""],input_names:[0,2,1,""],is_training:[0,2,1,""],output_names:[0,2,1,""],run:[0,3,1,""],save:[0,3,1,""],session:[0,2,1,""]},"cxflow_tensorflow.FrozenModel":{__init__:[0,3,1,""],input_names:[0,2,1,""],output_names:[0,2,1,""],restore_frozen_model:[0,4,1,""],run:[0,3,1,""],save:[0,3,1,""]},"cxflow_tensorflow.hooks":{DecayLR:[1,1,1,""],DecayLROnPlateau:[1,1,1,""],InitLR:[1,1,1,""],WriteTensorBoard:[1,1,1,""]},"cxflow_tensorflow.hooks.DecayLR":{LR_DECAY_TYPES:[1,2,1,""],__init__:[1,3,1,""],_decay_variable:[1,3,1,""],after_epoch:[1,3,1,""]},"cxflow_tensorflow.hooks.DecayLROnPlateau":{_on_plateau_action:[1,3,1,""]},"cxflow_tensorflow.hooks.InitLR":{__init__:[1,3,1,""],before_training:[1,3,1,""]},"cxflow_tensorflow.hooks.WriteTensorBoard":{MISSING_VARIABLE_ACTIONS:[1,2,1,""],UNKNOWN_TYPE_ACTIONS:[1,2,1,""],__init__:[1,3,1,""],after_epoch:[1,3,1,""]},"cxflow_tensorflow.metrics":{bin_dice:[2,5,1,""],bin_stats:[2,5,1,""]},"cxflow_tensorflow.models":{blocks:[4,0,0,"-"],cnn_autoencoder:[3,5,1,""],cnn_encoder:[3,5,1,""],conv:[5,0,0,"-"],conv_blocks:[6,0,0,"-"]},"cxflow_tensorflow.models.blocks":{BaseBlock:[4,1,1,""],Block:[4,1,1,""],UnrecognizedCodeError:[4,6,1,""],get_block_instance:[4,5,1,""]},"cxflow_tensorflow.models.blocks.BaseBlock":{__init__:[4,3,1,""],_handle_parsed_args:[4,3,1,""]},"cxflow_tensorflow.models.blocks.Block":{__init__:[4,3,1,""],apply:[4,3,1,""],inverse_code:[4,3,1,""]},"cxflow_tensorflow.models.conv":{CONV_BLOCKS:[5,7,1,""],POOL_BLOCKS:[5,7,1,""],UNPOOL_BLOCK:[5,2,1,""],cnn_autoencoder:[5,5,1,""],cnn_encoder:[5,5,1,""],compute_pool_amount:[5,5,1,""]},"cxflow_tensorflow.models.conv_blocks":{AveragePoolBlock:[6,1,1,""],ConvBaseBlock:[6,1,1,""],ConvBlock:[6,1,1,""],IncBlock:[6,1,1,""],MaxPoolBlock:[6,1,1,""],PoolBaseBlock:[6,1,1,""],ResBlock:[6,1,1,""],UnPoolBlock:[6,1,1,""]},"cxflow_tensorflow.models.conv_blocks.ConvBaseBlock":{__init__:[6,3,1,""]},"cxflow_tensorflow.models.conv_blocks.ConvBlock":{__init__:[6,3,1,""],_handle_parsed_args:[6,3,1,""]},"cxflow_tensorflow.models.conv_blocks.IncBlock":{__init__:[6,3,1,""]},"cxflow_tensorflow.models.conv_blocks.PoolBaseBlock":{__init__:[6,3,1,""]},"cxflow_tensorflow.models.conv_blocks.ResBlock":{__init__:[6,3,1,""]},"cxflow_tensorflow.models.conv_blocks.UnPoolBlock":{CODE_PREFIX:[6,2,1,""]},"cxflow_tensorflow.ops":{dense_to_sparse:[7,5,1,""],flatten3D:[7,5,1,""],repeat:[7,5,1,""],smooth_l1_loss:[7,5,1,""]},"cxflow_tensorflow.utils":{create_activation:[8,5,1,""],create_optimizer:[8,5,1,""]},cxflow_tensorflow:{BaseModel:[0,1,1,""],FrozenModel:[0,1,1,""],hooks:[1,0,0,"-"],metrics:[2,0,0,"-"],models:[3,0,0,"-"],ops:[7,0,0,"-"],utils:[8,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","staticmethod","Python static method"],"5":["py","function","Python function"],"6":["py","exception","Python exception"],"7":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:staticmethod","5":"py:function","6":"py:exception","7":"py:data"},terms:{"128inc":[3,5],"256inc":[3,5],"3s2":[3,5],"3x5x5":6,"512ress2":[3,5],"5s2":6,"64c3":[3,5,6],"64c3s2":[3,5,6],"64c7s2":[3,5],"abstract":[0,12],"case":0,"class":[2,8,12,14],"default":[0,1,4,6,14],"final":[10,14],"float":1,"function":[0,6,14],"import":14,"int":[0,1,6,7,14],"long":14,"new":[0,1,4,6,14],"return":[0,1,2,3,4,5,6,7,8],"s\u00f8rensen":2,"short":4,"static":0,"true":[0,1,3,5,12],"try":[4,6,14],"while":0,For:[1,3,5,13,14],One:[8,12],That:[12,14],The:[0,1,2,3,5,10,11,12,13,14],__init__:[0,1,4,6],_create_model:[0,13,14],_create_sess:0,_create_train_op:0,_decay_valu:1,_decay_vari:1,_handle_parsed_arg:[4,6],_initialize_vari:0,_lr:1,_on_plateau_act:1,_restore_checkpoint:0,_restore_model:0,_valu:1,_variabl:1,abil:13,about:1,abstractdataset:0,abstracthook:1,abstractmodel:0,access:13,accord:[0,8],accumulate_vari:1,accumulatevari:1,accuraci:[1,14],action:1,activ:[0,3,5,8,13,14],activation_nam:8,actual:0,adamoptim:14,add:[1,3,5,13],add_to_collect:13,added:[3,5],adding:13,addit:[0,1,4,14],addition:12,after:[0,1,3,5],after_epoch:1,again:[1,13],alia:5,all:[0,1,12,13,14],allow:[0,11],alow:12,alreadi:[0,12],also:13,altern:10,although:7,alwai:12,among:12,amount:[5,12],ani:[0,1,4,8,12,14],anoth:13,ap3s2:[3,5],ap_fn:6,api:[13,14],appear:13,append:[3,5],appli:[0,3,4,5,13,14],appropri:0,arbitrari:[12,14],architectur:[0,3,5,11,12],arg:[1,4],argmax:14,argument:[4,6,13],around:[3,5],assertionerror:[3,5],assum:6,attempt:[8,14],auto:[3,5],automat:[0,12],avail:[0,3,5,8,9],averag:[0,3,5,6,12],averagepoolblock:[5,6],axi:7,base:[0,1,3,4,5,6,11],baseblock:[4,6],baselin:14,basemodel:[0,1,13,14],basic:11,batch:[0,2,3,5,6,12],batch_norm:13,befor:[1,10,11,14],before_train:1,behavior:0,being:1,best:14,between:12,bin_dic:2,bin_stat:2,binari:2,bit:12,block:[0,3,5,6],block_kwarg:4,bn_fn:6,bn_kwarg:[3,5],bool:[0,1,3,5],both:[1,3,5,13,14],brief:13,bthwc:6,buffer:0,build:[3,5],burden:0,calcul:[2,7],call:1,callabl:[3,5,6,8],can:[0,3,4,5,10,11,12,13,14],candid:4,cannot:[3,5],carefulli:8,cast:[1,14],catchsigint:14,caus:13,certain:[0,13],chain:7,chanc:12,chang:[0,12,14],channel:[3,5,6],charact:6,check:[0,12],checkpoint:0,checkpoint_path:0,ckpt:0,classif:2,clone:10,cnn:[3,5,7],cnn_autoencod:[3,5],cnn_encod:[3,5],code:[0,3,4,5,6,11,14],code_prefix:6,coeffici:2,cognexa:[10,11],colleagu:14,collect:[0,1,13],com:[10,11],come:14,command:0,common:[13,14],compat:[0,14],complic:12,compon:12,comput:[0,1,2,5,12],compute_pool_amount:5,compute_stat:1,computestat:[1,14],concaten:12,concept:4,config:[0,3,5,8,12,14],configproto:0,configur:[0,3,4,5,12],connect:[3,5,14],consequ:12,construct:[0,3,5],contain:[0,14],contrari:14,contrib:14,control:14,conv1:14,conv2:14,conv2d:[0,13,14],conv:[0,3],conv_block:[0,3,5],conv_fn:6,conv_kwarg:[3,5],convbaseblock:6,convblock:[5,6],convert:7,convnet:14,convnetexampl:14,convolut:[3,5,6,14],correct:4,correctli:[3,5],cost:13,coupl:14,cpu:12,creat:[0,1,3,4,5,6,8,12,14],create_activ:8,create_optim:8,creation:13,cumbersom:12,current:0,custom:[0,7,14],cxf_is_train:0,cxflow:[0,1,4,9,10,12,13],cxflow_tensorflow:[9,13,14],cxtf:14,dai:12,data:[1,6,12,14],dataset:[0,12,14],decai:1,decay_lr:1,decay_typ:1,decay_valu:1,decaylr:1,decaylronplateau:1,decod:[3,5],deep:[12,13],def:[13,14],defin:[0,2,3,5,12,13,14],degrad:14,dens:[7,13,14],dense3:14,dense4:14,dense_s:14,dense_to_spars:7,depend:0,deriv:0,design:0,desir:1,detail:14,detect:1,determin:[0,7],dice:2,dict:[0,3,5,8],dictionari:0,differ:[0,12],digit:14,dim:[3,5,6],dimens:[6,7],dir:[0,1],directli:[10,14],directori:0,distinguish:13,distribut:12,dive:10,doc:8,document:14,doe:[3,5,14],don:14,done:[13,14],dropout:[0,14],dtype:[3,5],dure:[0,1,8,12,13],each:[0,3,4,5,12],easier:11,easili:14,effortlessli:11,either:[0,1,11],element:7,elig:12,elu:[3,5],emerg:13,empti:0,encod:[3,5],encoder_config:[3,5],enough:12,ensur:12,entri:8,epoch:1,epoch_data:1,epoch_id:1,equal:[12,14],error:1,essenti:11,etc:[0,3,5],eval:[0,3,5],evalu:13,even:[12,13],evenli:12,everi:[0,1],everyth:14,exactli:0,exampl:[0,3,5,6,12,14],execut:1,expand_dim:14,expect:[0,1,2,7,13,14],expens:12,experi:[12,14],explicit:13,expos:0,express:[4,6],extens:1,extra:6,extra_dim:6,extra_stream:14,fact:[0,12,13,14],fail:4,fals:[0,1,3,5],fast:[7,14],favorit:14,featur:[3,5],fed:13,feed:[0,12],feel:14,fetch:[0,12,14],few:14,file:[0,12,14],find:[0,8],fine:0,fit:13,flag:0,flatten3d:7,flatten:[7,14],float32:[3,5,14],flush:1,flush_sec:1,focu:11,focus:0,follow:[0,3,4,5,9,13,14],forward:[0,12,14],found:[12,14],framework:[12,14],free:14,freez:0,frequent:14,from:[0,1,2,3,4,5,10,12],frozen:0,frozen_model:0,frozenmodel:0,full:[0,12,14],fulli:14,gain:12,gener:13,get:4,get_block_inst:4,get_vari:[0,12],git:10,github:[10,11,12,14],given:[0,2,3,4,5,7,8],goal:0,gone:11,gpu:[0,11],gpu_opt:0,gradient:[0,12],graph:[0,1,8,12,13,14],graph_opt:0,graphkei:13,greater:[1,12,14],group:4,had:14,hand:14,handl:[0,4,6,8,12],happen:14,hard:14,has:[0,12],have:[11,13],heavi:14,height:[3,5],here:[3,5,14],hidden:0,hook:[0,9,12,14],hour:12,how:[13,14],howev:11,http:10,human:4,hundr:11,hyper:14,ident:[6,14],ignor:1,imag:[1,3,5,14],image_vari:1,imagin:14,implement:[0,14],implicitli:13,improv:13,inappropri:4,inc:[3,5],incblock:[5,6],incept:[3,5,6],inceptionnet:[3,5],includ:[0,1,3,5,7,14],incomplet:[0,12],incorpor:13,increas:12,indic:[3,5],infer:12,info:1,inform:13,inherit:4,init_lr:1,initi:[0,1],initlr:1,input:[0,3,4,5,7,13],input_nam:0,instal:11,instanc:[0,4],instead:[3,5,11,12],int64:14,integr:14,interv:1,introduc:[13,14],introduct:1,invers:[3,4,5],inverse_cod:4,invok:0,is_train:[0,3,5,13,14],issu:11,iter:[1,4],its:[0,7,12,14],itself:14,just:[0,14],kera:[13,14],kernel:[5,6],kernel_regular:13,kernel_s:[3,5,6],know:11,known:13,kwarg:[0,1,3,4,5,6],label:[2,14],larg:12,last:[0,1],layer:[0,3,5,6,13,14],learn:[1,12,13],learning_r:[0,1,8,14],least:8,less:12,let:[11,14],leverag:[11,12],lift:14,like:[4,6],line:[11,14],linear:1,list:[0,1,3,5,8,13,14],ln_fn:6,ln_kwarg:[3,5],load:0,locac:0,log:[0,1],log_dir:0,logit:14,logvari:14,long_term:1,loss:[0,1,7,13,14],loss_nam:0,lower:12,lr_decay_typ:1,lucid:13,luckili:[12,13],mai:[1,7,12,14],mail:11,main:0,main_loop:14,maintain:13,make:[10,11,12,14],manag:0,map:[0,1,4],mask:7,match:4,matter:14,max:[1,3,5,6],maxpool2d:14,maxpoolblock:[5,6],mean:[0,1,13,14],meet:[3,5],method:[0,4,13],metric:[0,9],might:10,minim:[0,14],minut:[11,14],miss:[1,14],missing_variable_act:1,mit:11,mnistdataset:14,model:[0,1,9,11],model_3:0,modifi:1,modul:[0,1,2,3,5,7,8,9],moment:[3,5],monitor:0,more:[1,12,14],most:[0,4,10,13,14],mp2:[3,5],mp_fn:6,much:[12,14],multi:[0,11],multipl:12,multipli:1,must:[0,4,8,14],mutabl:[3,5],my_dir:0,my_learning_r:1,my_loss:13,my_vari:1,mymultigpunet:12,n_gpu:[0,12],name:[0,1,2,3,5,8,13,14],name_suffix:0,natur:[12,13],necessari:12,need:[13,14],nest:0,net:[0,3,5,12,13,14],network:[0,3,5,13,14],neural:[3,5,14],neuron:14,never:11,non:[0,1,7],none:[0,1,2,3,4,5,6,7,14],nor:[3,5],normal:[0,3,5,6],note:14,noth:[12,14],now:[1,12],num_filt:[3,5,6],num_output:6,number:[0,6,7,14],object:[0,1,4],offici:11,often:[0,12],on_missing_vari:1,on_plateau:1,on_unknown_typ:1,onc:11,one:[0,1,14],ones:13,onli:[0,1,12,13,14],onplateau:1,oper:[0,3,5],ops:[0,9,13],optim:[0,8,13],optimizer_config:[0,8],option:[0,1,2,3,4,5,7,14],order:[0,3,4,5,12],origin:6,other:1,our:[10,11,12,14],output:[0,1,2,3,4,5,6,12,13],output_dir:1,output_nam:0,over:[12,13],overrid:0,overridden:4,own:[0,12],param:0,paramat:0,paramet:[0,1,2,3,4,5,6,7,8,12],parametr:[4,14],pars:[3,4,5,6],part:0,pass:[4,12,14],past:13,path:[0,14],perfect:12,perform:[7,12,13,14],perspect:12,phase:[3,5,13],piec:7,pip:10,placehold:[0,3,5,14],plateau:1,pleas:[1,11],plugin:11,point:0,pool:[3,5,6],pool_block:5,pool_fn:6,poolbaseblock:6,pop:[3,5],popular:13,portion:13,possibl:1,post:[3,5],potenti:[11,12],pre:[3,5],precis:[0,2],predict:[2,7,14],prefer:13,prefix:[0,2,6],prepar:14,present:[1,8],preserv:12,pretti:14,prevent:13,proce:11,procedur:[0,4],process:[3,5,8,12],produc:7,project:[11,14],properli:[10,14],properti:13,provid:[0,1,3,4,5,14],purpos:10,put:14,quit:[12,13,14],rais:[0,3,4,5],randomli:[0,13],rang:0,rate:1,read:[11,14],readabl:4,readi:12,reason:13,recal:2,recogn:[3,5,14],recommend:10,reduce_mean:14,refer:[0,1,3,5,7,14],referenc:14,regexp:4,regular:[4,6],regularization_loss:[0,13],regularization_losss:0,relu:[0,3,5],remain:12,render:12,repeat:7,repositori:[12,14],requir:[0,3,5,12],res:[3,5],resblock:[5,6],residu:[3,5,6],resnet:[3,5],resourc:[1,12],respect:[0,12,14],respons:14,restor:[0,12],restore_from:0,restore_frozen_model:0,restore_model_nam:0,resum:1,reus:12,revers:[3,5],rmspropoptim:0,routin:12,rtype:0,run:[0,12,13,14],said:[13,14],same:7,save:[0,1,4],saver:0,scalar:[1,13],scale:13,scope:[0,6],seamless:14,search:14,second:1,section:[0,14],see:[0,3,5,13,14],seem:7,select:13,self:[0,1,3,5,13,14],sequenc:[0,3,5],session:[0,8],session_config:0,set:[0,1,13,14],shall:4,shape:[3,5,7,14],share:[1,12],short_term:1,should:[0,14],show:13,signal:0,signal_mean:0,signal_mean_nam:0,signal_var_nam:0,signal_vari:0,significantli:12,similarli:14,simpl:[13,14],simpleconvnet:14,simplest:10,simpli:[0,13],singl:[0,4],size:[0,5,6,12],skip:[3,5],skip_connect:[3,5],slim:14,slow:12,small:[1,14],smooth:7,smooth_l1_loss:7,some:[0,3,5],sorensen:2,sourc:[0,1,2,3,4,5,6,7,8,10,11],source_nam:0,spars:7,sparse_softmax_cross_entropy_with_logit:14,sparsetensor:7,spatial:6,special:13,specifi:[0,1,3,5,7,14],speed:[11,12],srivastava:13,stand:0,start:11,statist:13,std:14,stopaft:14,str:[0,1,2,3,4,5,6,8],straightforward:12,stream:[0,14],streamwrapp:0,stride:[3,5,6],string:[1,4],sub:[0,13],suffix:[0,2],sum:[0,1],summari:1,support:[0,1,3,5],sure:[10,11,14],system:1,take:[1,4],taken:1,task:14,techniqu:13,ten:11,tensor:[0,2,3,4,5,6,7,8,13,14],tensorboard:[1,14],tensorflow:[0,4,8,9,10,12,13],test:14,tf_optimizer_modul:8,than:[1,12,14],thei:[0,3,5],them:13,therefor:14,thi:[0,1,3,5,7,8,12,13,14],those:[13,14],through:[0,11,13,14],time:[6,7,12],time_kernel:6,time_kernel_s:[3,5,6],togeth:[12,14],tower:[0,12],tracker:11,train:[0,1,3,5,11,13,14],train_op:0,train_op_1:0,train_op_2:0,train_op_nam:0,trainabl:0,training_flag_nam:0,troubl:12,tune:0,tupl:[2,3,4,5,6],tutori:[1,11,13],two:5,type:[0,1,2,3,4,5,6,7,8],unaffect:7,unavoid:14,under:[8,11,14],union:6,unit:13,unknown:1,unknown_type_act:1,unless:0,unpool:[3,5],unpool_block:5,unpoolblock:[5,6],unrecognizedcodeerror:[3,4,5],updat:[0,12,13,14],update_op:13,usag:[1,3,5],use:[0,3,5,7,12,14],use_bn:[3,5],use_ln:[3,5],used:[1,12,13],useful:[0,1],user:[0,10],using:[1,4,8,10],usual:12,util:[0,2,9,12,13,14],valid:1,valu:[0,1,2,4,7],valueerror:[0,3,5],vanilla:14,variabl:[0,1,8,12,14],variable_nam:1,variable_scop:14,varianc:[0,13],variou:[2,13],verbos:13,veri:13,via:[11,13],visual:1,visualize_graph:1,vt_co:4,wai:[0,10,12,13,14],warn:[1,7],weight:0,well:13,were:14,when:[1,3,5,8],where:14,wherein:0,whether:0,which:[0,4,12,14],whole:[12,14],width:[3,5],wise:7,without:12,work:[11,14],would:[5,14],wrapper:0,write:[1,11,14],write_tensorboard:1,writetensorboard:[1,14],written:14,wrongli:0,yaml:[0,12,14],year:13,yet:13,you:[11,13,14],your:[0,12,13,14],zero:[7,13]},titles:["<code class=\"docutils literal\"><span class=\"pre\">cxflow_tensorflow</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow_tensorflow.hooks</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow_tensorflow.metrics</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow_tensorflow.models</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow_tensorflow.models.blocks</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow_tensorflow.models.conv</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow_tensorflow.models.conv_blocks</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow_tensorflow.ops</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow_tensorflow.utils</span></code>","API Reference","Getting Started","Welcome to cxflow-tensorflow docs","Multi-GPU models","Model regularization","Tutorial"],titleterms:{"class":[0,1,4,5,6],"function":[2,3,4,5,7,8],api:9,basic:14,batch:13,block:4,compat:12,configur:14,contribut:11,conv:5,conv_block:6,cxflow:[11,14],cxflow_tensorflow:[0,1,2,3,4,5,6,7,8],decai:13,detail:12,develop:10,doc:11,dropout:13,except:4,first:14,get:10,gpu:12,hook:1,implement:12,input:14,instal:10,licens:11,metric:2,model:[3,4,5,6,12,13,14],multi:12,next:14,normal:13,ops:7,optim:14,output:14,paramet:14,refer:9,regular:13,start:10,step:14,submodul:[0,3],support:11,tensorflow:[11,14],train:12,tutori:14,util:8,variabl:5,weight:13,welcom:11}})