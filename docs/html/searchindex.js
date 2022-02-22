Search.setIndex({docnames:["index","modules","quaterion","quaterion.dataset","quaterion.dataset.indexing_dataset","quaterion.dataset.similarity_data_loader","quaterion.dataset.similarity_dataset","quaterion.dataset.similarity_samples","quaterion.dataset.train_collater","quaterion.eval","quaterion.eval.base_metric","quaterion.loss","quaterion.loss.arcface_loss","quaterion.loss.contrastive_loss","quaterion.loss.group_loss","quaterion.loss.metrics","quaterion.loss.pairwise_loss","quaterion.loss.similarity_loss","quaterion.loss.softmax_loss","quaterion.loss.triplet_loss","quaterion.main","quaterion.train","quaterion.train.trainable_model","quaterion.utils","quaterion.utils.enums","quaterion.utils.utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["index.rst","modules.rst","quaterion.rst","quaterion.dataset.rst","quaterion.dataset.indexing_dataset.rst","quaterion.dataset.similarity_data_loader.rst","quaterion.dataset.similarity_dataset.rst","quaterion.dataset.similarity_samples.rst","quaterion.dataset.train_collater.rst","quaterion.eval.rst","quaterion.eval.base_metric.rst","quaterion.loss.rst","quaterion.loss.arcface_loss.rst","quaterion.loss.contrastive_loss.rst","quaterion.loss.group_loss.rst","quaterion.loss.metrics.rst","quaterion.loss.pairwise_loss.rst","quaterion.loss.similarity_loss.rst","quaterion.loss.softmax_loss.rst","quaterion.loss.triplet_loss.rst","quaterion.main.rst","quaterion.train.rst","quaterion.train.trainable_model.rst","quaterion.utils.rst","quaterion.utils.enums.rst","quaterion.utils.utils.rst"],objects:{"":[[2,0,0,"-","quaterion"]],"quaterion.dataset":[[4,0,0,"-","indexing_dataset"],[5,0,0,"-","similarity_data_loader"],[6,0,0,"-","similarity_dataset"],[7,0,0,"-","similarity_samples"],[8,0,0,"-","train_collater"]],"quaterion.dataset.indexing_dataset":[[4,1,1,"","IndexingDataset"],[4,1,1,"","IndexingIterableDataset"]],"quaterion.dataset.indexing_dataset.IndexingIterableDataset":[[4,2,1,"","reinforce_type"]],"quaterion.dataset.similarity_data_loader":[[5,1,1,"","GroupSimilarityDataLoader"],[5,1,1,"","PairsSimilarityDataLoader"],[5,1,1,"","SimilarityDataLoader"]],"quaterion.dataset.similarity_data_loader.GroupSimilarityDataLoader":[[5,3,1,"","batch_size"],[5,2,1,"","collate_labels"],[5,3,1,"","dataset"],[5,3,1,"","drop_last"],[5,2,1,"","flatten_objects"],[5,3,1,"","num_workers"],[5,3,1,"","pin_memory"],[5,3,1,"","prefetch_factor"],[5,3,1,"","sampler"],[5,3,1,"","timeout"]],"quaterion.dataset.similarity_data_loader.PairsSimilarityDataLoader":[[5,3,1,"","batch_size"],[5,2,1,"","collate_labels"],[5,3,1,"","dataset"],[5,3,1,"","drop_last"],[5,2,1,"","flatten_objects"],[5,3,1,"","num_workers"],[5,3,1,"","pin_memory"],[5,3,1,"","prefetch_factor"],[5,3,1,"","sampler"],[5,3,1,"","timeout"]],"quaterion.dataset.similarity_data_loader.SimilarityDataLoader":[[5,3,1,"","batch_size"],[5,2,1,"","collate_labels"],[5,3,1,"","dataset"],[5,3,1,"","drop_last"],[5,2,1,"","flatten_objects"],[5,3,1,"","num_workers"],[5,4,1,"","original_params"],[5,3,1,"","pin_memory"],[5,2,1,"","pre_collate_fn"],[5,3,1,"","prefetch_factor"],[5,3,1,"","sampler"],[5,3,1,"","timeout"]],"quaterion.dataset.similarity_dataset":[[6,1,1,"","SimilarityGroupDataset"]],"quaterion.dataset.similarity_samples":[[7,1,1,"","SimilarityGroupSample"],[7,1,1,"","SimilarityPairSample"]],"quaterion.dataset.similarity_samples.SimilarityGroupSample":[[7,3,1,"","group"],[7,3,1,"","obj"]],"quaterion.dataset.similarity_samples.SimilarityPairSample":[[7,3,1,"","obj_a"],[7,3,1,"","obj_b"],[7,3,1,"","score"],[7,3,1,"","subgroup"]],"quaterion.dataset.train_collater":[[8,1,1,"","TrainCollater"]],"quaterion.dataset.train_collater.TrainCollater":[[8,2,1,"","pre_encoder_collate"]],"quaterion.eval":[[10,0,0,"-","base_metric"]],"quaterion.eval.base_metric":[[10,1,1,"","BaseMetric"]],"quaterion.eval.base_metric.BaseMetric":[[10,2,1,"","eval"]],"quaterion.loss":[[12,0,0,"-","arcface_loss"],[13,0,0,"-","contrastive_loss"],[14,0,0,"-","group_loss"],[15,0,0,"-","metrics"],[16,0,0,"-","pairwise_loss"],[17,0,0,"-","similarity_loss"],[18,0,0,"-","softmax_loss"],[19,0,0,"-","triplet_loss"]],"quaterion.loss.arcface_loss":[[12,1,1,"","ArcFaceLoss"]],"quaterion.loss.arcface_loss.ArcFaceLoss":[[12,2,1,"","forward"],[12,3,1,"","training"]],"quaterion.loss.contrastive_loss":[[13,1,1,"","ContrastiveLoss"],[13,5,1,"","relu"]],"quaterion.loss.contrastive_loss.ContrastiveLoss":[[13,2,1,"","forward"],[13,2,1,"","get_config_dict"],[13,2,1,"","metric_class"],[13,3,1,"","training"]],"quaterion.loss.group_loss":[[14,1,1,"","GroupLoss"]],"quaterion.loss.group_loss.GroupLoss":[[14,2,1,"","forward"],[14,3,1,"","training"]],"quaterion.loss.metrics":[[15,1,1,"","SiameseDistanceMetric"],[15,5,1,"","cosine_similarity"],[15,5,1,"","pairwise_distance"]],"quaterion.loss.metrics.SiameseDistanceMetric":[[15,2,1,"","cosine_distance"],[15,2,1,"","dot_product_distance"],[15,2,1,"","euclidean"],[15,2,1,"","manhattan"]],"quaterion.loss.pairwise_loss":[[16,1,1,"","PairwiseLoss"]],"quaterion.loss.pairwise_loss.PairwiseLoss":[[16,2,1,"","forward"],[16,3,1,"","training"]],"quaterion.loss.similarity_loss":[[17,1,1,"","SimilarityLoss"]],"quaterion.loss.similarity_loss.SimilarityLoss":[[17,2,1,"","get_config_dict"],[17,2,1,"","get_distance_function"],[17,2,1,"","metric_class"],[17,3,1,"","training"]],"quaterion.loss.softmax_loss":[[18,1,1,"","SoftmaxLoss"]],"quaterion.loss.softmax_loss.SoftmaxLoss":[[18,2,1,"","forward"],[18,3,1,"","training"]],"quaterion.loss.triplet_loss":[[19,1,1,"","TripletLoss"]],"quaterion.loss.triplet_loss.TripletLoss":[[19,2,1,"","forward"],[19,2,1,"","get_config_dict"],[19,3,1,"","training"]],"quaterion.main":[[20,1,1,"","Quaterion"]],"quaterion.main.Quaterion":[[20,2,1,"","fit"]],"quaterion.train":[[22,0,0,"-","trainable_model"]],"quaterion.train.trainable_model":[[22,1,1,"","TrainableModel"]],"quaterion.train.trainable_model.TrainableModel":[[22,2,1,"","cache"],[22,2,1,"","configure_caches"],[22,2,1,"","configure_encoders"],[22,2,1,"","configure_head"],[22,2,1,"","configure_loss"],[22,4,1,"","loss"],[22,4,1,"","model"],[22,2,1,"","process_results"],[22,2,1,"","save_servable"],[22,2,1,"","setup_dataloader"],[22,2,1,"","test_step"],[22,3,1,"","training"],[22,2,1,"","training_step"],[22,2,1,"","validation_step"]],"quaterion.utils":[[24,0,0,"-","enums"],[25,0,0,"-","utils"]],"quaterion.utils.enums":[[24,1,1,"","TrainStage"]],"quaterion.utils.enums.TrainStage":[[24,3,1,"","TEST"],[24,3,1,"","TRAIN"],[24,3,1,"","VALIDATION"]],"quaterion.utils.utils":[[25,5,1,"","info_value_of_dtype"],[25,5,1,"","max_value_of_dtype"],[25,5,1,"","min_value_of_dtype"]],quaterion:[[3,0,0,"-","dataset"],[9,0,0,"-","eval"],[11,0,0,"-","loss"],[20,0,0,"-","main"],[21,0,0,"-","train"],[23,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","property","Python property"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:property","5":"py:function"},terms:{"0":[5,7,12,13,18,19],"03832":19,"05":18,"06":13,"07698":12,"1":[5,13,15],"10":7,"100":15,"11":7,"128":15,"1503":19,"1801":12,"1e":15,"1st_pair_1st_obj":5,"1st_pair_2nd_obj":5,"2":[5,16],"209":7,"2nd_pair_1st_obj":5,"2nd_pair_2nd_obj":5,"3":5,"371":7,"5":[12,19],"555":7,"6":7,"64":12,"7":7,"8":[7,15],"9":7,"case":7,"class":[4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,22,24],"default":[8,15,18,19,22],"do":22,"enum":[2,23],"float":[5,7,12,13,15,18,19,25],"function":[5,13,14,15,16,17,19,22],"int":[5,7,8,12,15,18,22,25],"new":5,"return":[5,12,13,14,15,16,17,18,19,22,25],"static":15,"true":[13,15,19],A:20,And:4,If:[13,15],In:7,It:[18,19],One:19,The:[13,14,16,17],_2:15,_s:5,ab:[12,19],about:25,actual:5,ad:22,addit:[5,12,13,22],addition:5,all:[7,15,19,22],allow:25,along:15,an:[18,22],angular:12,ani:[4,5,7,8,13,17,19,22],anoth:7,apart:[12,19],appl:5,appli:[5,22],ar:[5,13],arcface_loss:[2,11],arcfaceloss:12,arg:22,argument:[5,13,22],arxiv:[12,19],assembl:20,assign:[15,22],associ:[5,12,14,18,19,22],associat:5,attribut:17,auto:22,automat:22,avail:[13,17],averag:13,avoid:15,bach:5,bar:22,barak_obama_1:7,barak_obama_2:7,barak_obama_3:7,base:[4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,22,24],base_metr:[2,9],basemetr:10,batch:[5,8,13,16,19,22],batch_idx:22,batch_siz:[5,12,14,15,16,18,19,22],befor:[5,22],between:[13,14,15,16,17,22],bool:[5,12,13,14,15,16,17,18,19,22,25],broadcast:15,cach:[20,22],cachabl:22,cache_typ:22,cacheconfig:22,cachemixin:22,cachetyp:22,calcul:[15,17,19,22],call:13,callabl:[8,17],can:[13,14,16,17],candi:7,cannot:13,cdot:15,checkpoint:22,cheesecak:7,chopra:13,chosen:15,cl:17,classmethod:[5,13,17,20],closer:7,collat:[5,22],collate_fn:5,collate_label:5,com:13,common:15,comput:[12,13,15,16,18,22],config:[13,17,19],configur:22,configure_cach:22,configure_encod:22,configure_head:22,configure_loss:22,consum:5,contain:[13,14,16,17],content:1,contrast:13,contrastive_loss:[2,11],contrastiveloss:13,convert:5,correspond:[13,22],cosin:15,cosine_dist:[13,14,15,16,17],cosine_similar:15,cpu:22,cross:[12,18],csv:7,current:[13,17],data:[5,7,20,22,25],dataload:[5,20,22],datapip:4,dataset:[1,2,12,18,22],default_encoder_kei:22,defin:[12,13,14,16,17,19,22],design:18,dfrac:15,dict:[5,8,13,17,19,22],differ:7,dim:15,dimens:[12,15,18],directli:5,disambigu:22,displai:22,distanc:[13,14,15,16,17,19],distance_metric_nam:[13,14,16,17,19],distinguish:13,divid:18,divis:15,doe:25,dot:[15,18],dot_product_dist:15,drop_last:5,dtype:25,dure:[20,22],dwarf:20,e:22,each:[5,7,22],effect:15,either:13,elon_musk_1:7,elon_musk_2:7,elon_musk_3:7,embed:[12,13,14,16,17,18,19,22],embedding_dim:19,embedding_s:[12,18,22],encod:[5,8,12,18,22],encoder_col:8,encoder_nam:8,encoderhead:22,encor:5,entir:15,entropi:[12,18],enumer:5,ep:15,epsilon:15,especi:15,estim:5,euclidean:[15,19],eval:[1,2],evalu:22,exampl:[5,7,13,15,19,22],exdb:13,expect:13,expected_typ:4,f:15,face:7,fals:15,farther:20,featir:5,featur:[5,8],fewer:15,file_nam:7,fill:22,finfo:25,first:[7,13,15],fit:20,flat:15,flatten_object:5,forward:[12,13,14,16,18,19],from:[5,7,12,17,18,22],function_nam:17,further:[7,13],g:22,gener:[5,13],get:22,get_config_dict:[13,17,19],get_distance_funct:17,giant:20,given:25,gpu:22,grapefruit:7,group:[5,7,12,14,18,19],group_id:7,group_loss:[2,11],grouploss:[12,14,18,19],groupsimilaritydataload:5,ha:[13,17,19],hadsel:13,half:13,handl:[20,24],hard:19,hash:22,hash_id:5,have:[7,13,15,22],head:22,hint:4,how:15,http:[12,13,19],id:[5,7,8,12,16,18,22],iinfo:25,image_encod:22,implement:[8,18,19],increas:13,independ:22,index:[0,22],indexing_dataset:[2,3],indexingdataset:4,indexingiterabledataset:4,indic:13,individu:[5,22],info:25,info_value_of_dtyp:25,inform:13,initi:[5,22],input1:15,input2:15,input:[5,13,15],input_embedding_s:22,instanc:[4,17,20,22],integ:22,intern:20,item:[5,22],iterabledataset:4,itself:15,jpg:7,json:[13,17,19],kei:[13,22],key_extractor:22,keyword:22,kind:22,known:16,kwarg:[5,13,22],label:[5,13,16,19],labels_batch:5,layer:22,learn:7,least:13,lecun:13,lemon:[5,7],leonard_nimoy_1:7,leonard_nimoy_2:7,lightn:22,lightningmodul:22,lime:7,list:[5,8,16,22],load:[13,17,19,22],loader:[20,22],logger:22,logit:18,longtensor:[13,14,18,19],loss:[1,2,5,22],macaroon:7,mai:15,main:[1,2],make:[12,15],mandarin:7,manhattan:15,map:22,margin:[12,13,19],match:7,matrix:15,max:[15,25],max_value_of_dtyp:25,maximum:25,method:22,metric:[2,5,11,13,14,16,17,22],metric_class:[13,17],metricmodel:22,might:[8,22],min:25,min_value_of_dtyp:25,mine:19,mini:13,minimum:25,model:[20,22],modul:[0,1],more:[5,22],muffin:7,must:15,name:[13,14,16,17,19,22],neg:[13,19],nn:7,non:22,none:[8,15,20,22],num_group:[12,18],num_work:5,number:[12,18,22],nutella:7,obj:[5,7],obj_a:[5,7],obj_b:[5,7],object:[5,7,8,10,13,15,16,17,19,20,22,25],offset:5,one:[5,7,22],onli:5,onlin:19,option:[5,8,15,19,20,22],orang:[5,7],org:[12,19],origin:[4,5,22],original_param:5,other:7,otherwis:[15,22],output:[12,15,18,22],overridden:8,packag:1,page:0,pair:[5,13,16],pairs_count:16,pairssimilaritydataload:5,pairwis:16,pairwise_dist:15,pairwise_loss:[2,11],pairwiseloss:[13,16],param:[5,13,17,19],paramet:[5,12,13,14,15,16,17,18,19,20,22,25],pass:[5,17,22,25],path:22,pdf:13,per:8,perform:20,person:7,pictur:7,pin_memori:5,posit:13,pre:[13,14,16,17],pre_collate_fn:[5,8],pre_encoder_col:8,predict:5,prefetch_factor:5,prepar:8,print:15,process:[20,22],process_result:22,produc:22,product:[15,18],progress:22,promot:15,properti:[5,22],provid:22,pseudo:5,publi:13,purpos:[13,17,19],push:[12,19],pytorch:[22,25],pytorch_lightn:20,quaterion_model:22,queri:7,rais:[17,25],ram:22,randn:15,random:5,raw:5,reduc:13,refer:15,regular:18,reinforc:4,reinforce_typ:4,relu:13,repres:7,requir:4,respect:5,restrict:4,result:15,retriev:[5,17,20],routin:20,runtimeerror:17,s:[20,22],sampl:[5,13,20],sampler:5,save:[13,17,19,22],save_serv:22,scalar:19,scale:12,score:[5,7,13],search:0,second:[7,13,15],see:[15,20],serializ:[13,17,19],serv:22,setup_dataload:22,shape:[12,14,15,16,18,19,22],should:[5,7,13,22],shoulder:20,siamesedistancemetr:[13,14,15,16,17],similar:[5,7,15,16,17],similarity_data_load:[2,3],similarity_dataset:[2,3],similarity_loss:[2,11],similarity_sampl:[2,3],similaritydataload:[5,20,22],similaritygroupdataset:6,similaritygroupsampl:[5,6,7],similarityloss:[14,16,17,22],similaritypairsampl:[5,7],singl:7,size:[13,14,16,18,22],size_averag:13,small:15,softmax:18,softmax_loss:[2,11],softmaxloss:18,some:22,sophist:22,sourc:[4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,22,24,25],specif:[5,22],specifi:[19,22],split:5,squar:[15,19],squeez:15,stage:[20,22,24],state:22,store:22,str:[5,8,13,14,16,17,19,22,24],strategi:19,subgroup:[5,7,13,16],submodul:1,subpackag:1,subtyp:4,suitabl:5,sum:13,support:[15,19],sure:15,t_co:[4,5],target:[5,22],temperatur:18,tensor:[5,8,10,12,13,14,15,16,17,18,19,22],test:[22,24],test_step:22,text:[13,15],text_encod:22,than:5,thei:22,thi:[5,7,15,22],timeout:5,torch:[5,15,19,25],train:[1,2,12,13,14,16,17,18,19,20,24],train_collat:[2,3],train_dataload:[20,22],trainabl:22,trainable_model:[2,20,21],trainablemodel:[20,22],traincollat:8,trainer:[20,22],training_step:22,trainstag:[22,24],transform:5,triplet:19,triplet_loss:[2,11],tripletloss:19,tupl:[4,5,8,22],two:[13,14,16,17,20],type:[4,13,15,17,22,25],typeerror:25,understand:15,unexpect:15,union:[8,22,25],uniqu:[5,7],unknown:17,updat:22,us:[5,13,14,16,17,19,22],util:[1,2,5],val_dataload:[20,22],valid:[20,22,24],validation_step:22,valu:[12,13,14,15,16,18,19,22,24,25],vector:22,vector_length:[12,14,16,18],vert:15,wai:22,want:22,well:22,what:22,which:[7,13,15,22,25],whole:20,within:7,without:5,word:13,work:[12,15,18],x1:15,x2:15,x:15,x_1:15,x_2:15,y:15,yann:13,you:[15,22],zero:[14,15,16,18]},titles:["Welcome to Quaterion\u2019s documentation!","quaterion","quaterion package","quaterion.dataset package","quaterion.dataset.indexing_dataset module","quaterion.dataset.similarity_data_loader module","quaterion.dataset.similarity_dataset module","quaterion.dataset.similarity_samples module","quaterion.dataset.train_collater module","quaterion.eval package","quaterion.eval.base_metric module","quaterion.loss package","quaterion.loss.arcface_loss module","quaterion.loss.contrastive_loss module","quaterion.loss.group_loss module","quaterion.loss.metrics module","quaterion.loss.pairwise_loss module","quaterion.loss.similarity_loss module","quaterion.loss.softmax_loss module","quaterion.loss.triplet_loss module","quaterion.main module","quaterion.train package","quaterion.train.trainable_model module","quaterion.utils package","quaterion.utils.enums module","quaterion.utils.utils module"],titleterms:{"enum":24,arcface_loss:12,base_metr:10,content:[2,3,9,11,21,23],contrastive_loss:13,dataset:[3,4,5,6,7,8],document:0,eval:[9,10],group_loss:14,indexing_dataset:4,indic:0,loss:[11,12,13,14,15,16,17,18,19],main:20,metric:15,modul:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],packag:[2,3,9,11,21,23],pairwise_loss:16,quaterion:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],s:0,similarity_data_load:5,similarity_dataset:6,similarity_loss:17,similarity_sampl:7,softmax_loss:18,submodul:[2,3,9,11,21,23],subpackag:[2,21],tabl:0,train:[21,22],train_collat:8,trainable_model:22,triplet_loss:19,util:[23,24,25],welcom:0}})