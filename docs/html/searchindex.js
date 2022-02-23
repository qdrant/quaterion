Search.setIndex({docnames:["index","modules","quaterion","quaterion.dataset","quaterion.dataset.indexing_dataset","quaterion.dataset.similarity_data_loader","quaterion.dataset.similarity_dataset","quaterion.dataset.similarity_samples","quaterion.dataset.train_collater","quaterion.eval","quaterion.eval.base_metric","quaterion.loss","quaterion.loss.arcface_loss","quaterion.loss.contrastive_loss","quaterion.loss.group_loss","quaterion.loss.metrics","quaterion.loss.pairwise_loss","quaterion.loss.similarity_loss","quaterion.loss.softmax_loss","quaterion.loss.triplet_loss","quaterion.main","quaterion.train","quaterion.train.cache","quaterion.train.cache.cache_config","quaterion.train.trainable_model","quaterion.utils","quaterion.utils.enums","quaterion.utils.utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["index.rst","modules.rst","quaterion.rst","quaterion.dataset.rst","quaterion.dataset.indexing_dataset.rst","quaterion.dataset.similarity_data_loader.rst","quaterion.dataset.similarity_dataset.rst","quaterion.dataset.similarity_samples.rst","quaterion.dataset.train_collater.rst","quaterion.eval.rst","quaterion.eval.base_metric.rst","quaterion.loss.rst","quaterion.loss.arcface_loss.rst","quaterion.loss.contrastive_loss.rst","quaterion.loss.group_loss.rst","quaterion.loss.metrics.rst","quaterion.loss.pairwise_loss.rst","quaterion.loss.similarity_loss.rst","quaterion.loss.softmax_loss.rst","quaterion.loss.triplet_loss.rst","quaterion.main.rst","quaterion.train.rst","quaterion.train.cache.rst","quaterion.train.cache.cache_config.rst","quaterion.train.trainable_model.rst","quaterion.utils.rst","quaterion.utils.enums.rst","quaterion.utils.utils.rst"],objects:{"":[[2,0,0,"-","quaterion"]],"quaterion.dataset":[[4,0,0,"-","indexing_dataset"],[5,0,0,"-","similarity_data_loader"],[6,0,0,"-","similarity_dataset"],[7,0,0,"-","similarity_samples"],[8,0,0,"-","train_collater"]],"quaterion.dataset.indexing_dataset":[[4,1,1,"","IndexingDataset"],[4,1,1,"","IndexingIterableDataset"]],"quaterion.dataset.indexing_dataset.IndexingIterableDataset":[[4,2,1,"","reinforce_type"]],"quaterion.dataset.similarity_data_loader":[[5,1,1,"","GroupSimilarityDataLoader"],[5,1,1,"","PairsSimilarityDataLoader"],[5,1,1,"","SimilarityDataLoader"]],"quaterion.dataset.similarity_data_loader.GroupSimilarityDataLoader":[[5,3,1,"","batch_size"],[5,2,1,"","collate_labels"],[5,3,1,"","dataset"],[5,3,1,"","drop_last"],[5,2,1,"","flatten_objects"],[5,3,1,"","num_workers"],[5,3,1,"","pin_memory"],[5,3,1,"","prefetch_factor"],[5,3,1,"","sampler"],[5,3,1,"","timeout"]],"quaterion.dataset.similarity_data_loader.PairsSimilarityDataLoader":[[5,3,1,"","batch_size"],[5,2,1,"","collate_labels"],[5,3,1,"","dataset"],[5,3,1,"","drop_last"],[5,2,1,"","flatten_objects"],[5,3,1,"","num_workers"],[5,3,1,"","pin_memory"],[5,3,1,"","prefetch_factor"],[5,3,1,"","sampler"],[5,3,1,"","timeout"]],"quaterion.dataset.similarity_data_loader.SimilarityDataLoader":[[5,3,1,"","batch_size"],[5,2,1,"","collate_labels"],[5,3,1,"","dataset"],[5,3,1,"","drop_last"],[5,2,1,"","flatten_objects"],[5,3,1,"","num_workers"],[5,4,1,"","original_params"],[5,3,1,"","pin_memory"],[5,2,1,"","pre_collate_fn"],[5,3,1,"","prefetch_factor"],[5,3,1,"","sampler"],[5,3,1,"","timeout"]],"quaterion.dataset.similarity_dataset":[[6,1,1,"","SimilarityGroupDataset"]],"quaterion.dataset.similarity_samples":[[7,1,1,"","SimilarityGroupSample"],[7,1,1,"","SimilarityPairSample"]],"quaterion.dataset.similarity_samples.SimilarityGroupSample":[[7,3,1,"","group"],[7,3,1,"","obj"]],"quaterion.dataset.similarity_samples.SimilarityPairSample":[[7,3,1,"","obj_a"],[7,3,1,"","obj_b"],[7,3,1,"","score"],[7,3,1,"","subgroup"]],"quaterion.dataset.train_collater":[[8,1,1,"","TrainCollater"]],"quaterion.dataset.train_collater.TrainCollater":[[8,2,1,"","pre_encoder_collate"]],"quaterion.eval":[[10,0,0,"-","base_metric"]],"quaterion.eval.base_metric":[[10,1,1,"","BaseMetric"]],"quaterion.eval.base_metric.BaseMetric":[[10,2,1,"","eval"]],"quaterion.loss":[[12,0,0,"-","arcface_loss"],[13,0,0,"-","contrastive_loss"],[14,0,0,"-","group_loss"],[15,0,0,"-","metrics"],[16,0,0,"-","pairwise_loss"],[17,0,0,"-","similarity_loss"],[18,0,0,"-","softmax_loss"],[19,0,0,"-","triplet_loss"]],"quaterion.loss.arcface_loss":[[12,1,1,"","ArcFaceLoss"],[12,5,1,"","l2_norm"]],"quaterion.loss.arcface_loss.ArcFaceLoss":[[12,2,1,"","forward"],[12,3,1,"","training"]],"quaterion.loss.contrastive_loss":[[13,1,1,"","ContrastiveLoss"]],"quaterion.loss.contrastive_loss.ContrastiveLoss":[[13,2,1,"","forward"],[13,2,1,"","get_config_dict"],[13,2,1,"","metric_class"],[13,3,1,"","training"]],"quaterion.loss.group_loss":[[14,1,1,"","GroupLoss"]],"quaterion.loss.group_loss.GroupLoss":[[14,2,1,"","forward"],[14,3,1,"","training"]],"quaterion.loss.metrics":[[15,1,1,"","SiameseDistanceMetric"]],"quaterion.loss.metrics.SiameseDistanceMetric":[[15,2,1,"","cosine_distance"],[15,2,1,"","dot_product_distance"],[15,2,1,"","euclidean"],[15,2,1,"","manhattan"]],"quaterion.loss.pairwise_loss":[[16,1,1,"","PairwiseLoss"]],"quaterion.loss.pairwise_loss.PairwiseLoss":[[16,2,1,"","forward"],[16,3,1,"","training"]],"quaterion.loss.similarity_loss":[[17,1,1,"","SimilarityLoss"]],"quaterion.loss.similarity_loss.SimilarityLoss":[[17,2,1,"","get_config_dict"],[17,2,1,"","get_distance_function"],[17,2,1,"","metric_class"],[17,3,1,"","training"]],"quaterion.loss.softmax_loss":[[18,1,1,"","SoftmaxLoss"]],"quaterion.loss.softmax_loss.SoftmaxLoss":[[18,2,1,"","forward"],[18,3,1,"","training"]],"quaterion.loss.triplet_loss":[[19,1,1,"","TripletLoss"]],"quaterion.loss.triplet_loss.TripletLoss":[[19,2,1,"","forward"],[19,2,1,"","get_config_dict"],[19,3,1,"","training"]],"quaterion.main":[[20,1,1,"","Quaterion"]],"quaterion.main.Quaterion":[[20,2,1,"","fit"]],"quaterion.train":[[22,0,0,"-","cache"],[24,0,0,"-","trainable_model"]],"quaterion.train.cache":[[22,1,1,"","CacheConfig"],[22,1,1,"","CacheType"],[23,0,0,"-","cache_config"]],"quaterion.train.cache.CacheConfig":[[22,3,1,"","batch_size"],[22,3,1,"","cache_type"],[22,3,1,"","key_extractors"],[22,3,1,"","mapping"],[22,3,1,"","num_workers"]],"quaterion.train.cache.CacheType":[[22,3,1,"","AUTO"],[22,3,1,"","CPU"],[22,3,1,"","GPU"]],"quaterion.train.cache.cache_config":[[23,1,1,"","CacheConfig"],[23,1,1,"","CacheType"]],"quaterion.train.cache.cache_config.CacheConfig":[[23,3,1,"","batch_size"],[23,3,1,"","cache_type"],[23,3,1,"","key_extractors"],[23,3,1,"","mapping"],[23,3,1,"","num_workers"]],"quaterion.train.cache.cache_config.CacheType":[[23,3,1,"","AUTO"],[23,3,1,"","CPU"],[23,3,1,"","GPU"]],"quaterion.train.trainable_model":[[24,1,1,"","TrainableModel"]],"quaterion.train.trainable_model.TrainableModel":[[24,2,1,"","cache"],[24,2,1,"","configure_caches"],[24,2,1,"","configure_encoders"],[24,2,1,"","configure_head"],[24,2,1,"","configure_loss"],[24,4,1,"","loss"],[24,4,1,"","model"],[24,2,1,"","process_results"],[24,2,1,"","save_servable"],[24,2,1,"","setup_dataloader"],[24,2,1,"","test_step"],[24,3,1,"","training"],[24,2,1,"","training_step"],[24,2,1,"","validation_step"]],"quaterion.utils":[[26,0,0,"-","enums"],[27,0,0,"-","utils"]],"quaterion.utils.enums":[[26,1,1,"","TrainStage"]],"quaterion.utils.enums.TrainStage":[[26,3,1,"","TEST"],[26,3,1,"","TRAIN"],[26,3,1,"","VALIDATION"]],"quaterion.utils.utils":[[27,5,1,"","info_value_of_dtype"],[27,5,1,"","max_value_of_dtype"],[27,5,1,"","min_value_of_dtype"]],quaterion:[[3,0,0,"-","dataset"],[9,0,0,"-","eval"],[11,0,0,"-","loss"],[20,0,0,"-","main"],[21,0,0,"-","train"],[25,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","property","Python property"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:property","5":"py:function"},terms:{"0":[5,7,12,13,18,19],"03832":19,"05":18,"06":13,"07698":12,"1":[5,7,13,15],"10":7,"11":7,"1503":19,"1801":12,"1st_pair_1st_obj":5,"1st_pair_2nd_obj":5,"2":[5,7,16],"209":7,"2nd_pair_1st_obj":5,"2nd_pair_2nd_obj":5,"3":[5,7],"32":[22,23],"4":7,"5":[12,19],"555":7,"64":12,"7":7,"8":7,"9":7,"case":7,"class":[4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,22,23,24,26],"default":[8,18,19,24],"do":24,"enum":[2,22,23,25],"float":[5,7,12,13,18,19,27],"function":[5,8,13,14,15,16,17,19,22,23,24],"int":[5,7,8,12,18,22,23,24,27],"new":5,"return":[5,6,12,13,14,15,16,17,18,19,24,27],"static":15,"true":[13,15,19],A:20,And:4,For:[],If:[13,15],In:7,It:[18,19,22,23],One:19,The:[13,14,16,17],There:[],__hash__:[],__init__:5,_s:5,ab:[12,19],abl:[],about:27,abov:[],actual:5,ad:24,add:[],addit:[5,12,13,24],addition:5,affect:[22,23],after:[],aggreg:8,all:[7,8,15,19,24],allow:27,along:[],alreadi:[],also:[],among:8,an:[18,24],angular:12,ani:[4,5,7,8,13,17,19,22,23,24],anoth:7,apart:[12,19],appl:5,appli:[5,12,24],ar:[5,13],arbitrari:[],arcface_loss:[2,11],arcfaceloss:12,arg:24,argument:[5,13,24],arxiv:[12,19],assembl:20,assign:[5,15,24],associ:[5,12,14,18,19,24],associat:5,attribut:17,auto:[22,23,24],automat:[5,24],avail:[13,17,22,23],averag:13,avoid:[],bach:5,bar:24,base:[4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,22,23,24,26],base_metr:[2,9],basemetr:10,batch:[5,8,13,16,19,22,23,24],batch_cifar:[],batch_idx:24,batch_mnist:[],batch_siz:[5,12,14,15,16,18,19,22,23,24],becaus:[],befor:[5,24],between:[13,14,15,16,17,24],bool:[5,12,13,14,15,16,17,18,19,24,27],cach:[2,20,21,24],cachabl:24,cachable_encod:[],cache_col:[],cache_config:[21,22],cache_encod:[],cache_mixin:[],cache_model:[],cache_multiprocessing_context:[],cache_train_collat:[],cache_typ:[22,23,24],cacheabl:[22,23],cachecollatefntyp:[],cacheconfig:[22,23,24],cachedataload:[22,23],cachedencod:[],cacheencod:[],cachemixin:24,cachemod:[],cachemodel:[],cachetraincollat:[],cachetyp:[22,23,24],calcul:[15,17,19,24],call:13,callabl:[8,17,22,23],can:[13,14,16,17],candi:7,cannot:13,checkpoint:24,cheesecak:7,chopra:13,chosen:15,cifar:[],cifar_load:[],cl:17,classif:6,classmethod:[5,13,17,20],closer:7,collat:[5,8,24],collate_fn:5,collate_label:5,collect:[],com:13,common:15,compat:6,compos:[],comput:[12,13,15,16,18,24],config:[13,17,19],configur:24,configure_cach:[22,23,24],configure_encod:24,configure_head:24,configure_loss:24,consum:5,contain:[13,14,16,17],content:1,contrast:13,contrastive_loss:[2,11],contrastiveloss:13,conveni:[],convert:[5,6],correct:[],correspond:13,cosin:15,cosine_dist:[13,14,15,16,17],cpu:[22,23,24],cross:[12,18],cuda:[22,23],current:[13,17],custom:[],data:[5,6,7,20,24,27],dataload:[5,20,24],dataloader_idx:[],datapip:4,dataset:[1,2,12,18,24],debug:5,def:[],default_encoder_kei:24,defin:[5,12,13,14,16,17,19,24],design:18,determinist:[],devic:[22,23],dict:[5,8,13,17,19,22,23,24],differ:7,dim:12,dimens:[12,18],directli:5,directori:[],disambigu:24,displai:24,distanc:[13,14,15,16,17,19],distance_metric_nam:[13,14,16,17,19],distinguish:13,distribut:[],divid:18,doe:[22,23,27],don:[],dot:[15,18],dot_product_dist:15,download:[],drop_last:5,dtype:27,dummi:5,dure:[20,22,23,24],dwarf:20,e:24,each:[5,7,24],effect:15,either:13,elon_musk_1:7,elon_musk_2:7,elon_musk_3:7,els:[22,23],embed:[12,13,14,16,17,18,19,24],embedding_dim:19,embedding_s:[12,18,24],encod:[5,8,12,18,22,23,24],encoder_col:8,encoder_nam:[8,22,23],encoderhead:24,encor:5,entir:15,entropi:[12,18],enumer:5,especi:15,estim:5,euclidean:[15,19],eval:[1,2],evalu:24,exampl:[5,7,13,19,24],exdb:13,expect:13,expected_typ:4,extract_kei:[],extractor:[22,23],face:7,factori:[22,23],fals:15,farther:20,featir:5,featur:[5,8],file_nam:7,fill:24,fill_cach:[],finfo:27,first:[7,13],fit:20,flag:[],flat:15,flatten_object:5,follow:[],fork:[],format:6,forward:[12,13,14,16,18,19],from:[5,7,12,17,18,24],frozen:[],function_nam:17,further:[7,13],g:24,gener:[5,13],genuin:[],get:24,get_collate_fn:5,get_config_dict:[13,17,19],get_distance_funct:17,giant:20,given:27,gpu:[22,23,24],group:[5,7,12,14,18,19],group_id:7,group_loss:[2,11],grouploss:[12,14,18,19],groupsimilaritydataload:[5,6],ha:[13,17,19],hadsel:13,half:13,handl:[20,26],happen:[],hard:19,hardwar:[],hash:24,hash_id:5,hashabl:[22,23],have:[7,13,15],head:24,here:[],hint:4,how:15,howev:[],http:[12,13,19],id:[5,7,8,12,16,18,24],iinfo:27,image_encod:24,implement:[8,18,19],in_memory_cache_encod:[],increas:13,independ:24,index:[0,24],indexing_dataset:[2,3],indexingdataset:4,indexingiterabledataset:4,indic:13,individu:[5,24],infer:[],info:27,info_value_of_dtyp:27,inform:[8,13],inherit:[],initi:[5,24],inmemorycacheencod:[],input:[5,12,13],input_embedding_s:24,input_path:[],instanc:[4,17,20,24],instanti:[],instead:[],integ:24,intern:20,item:[5,24],iterabledataset:4,itself:15,jpg:7,json:[13,17,19],kei:[13,22,23,24],key_extractor:[22,23,24],keyword:24,kind:24,known:16,kwarg:[5,13,24],l2:12,l2_norm:12,label:[5,6,13,16,19],labels_batch:5,layer:24,learn:7,least:13,lecun:13,lemon:[5,7],leonard_nimoy_1:7,leonard_nimoy_2:7,lightn:24,lightningmodul:24,lime:7,list:[5,8,16,24],load:[13,17,19,24],loader:[20,24],loader_a:[],loader_b:[],loader_n:[],logger:24,logit:18,longtensor:[13,14,18,19],loss:[1,2,5,24],macaroon:7,mai:15,main:[1,2],make:[12,15],manag:[],manhattan:15,map:[22,23,24],margin:[12,13,19],match:7,matrix:15,max:27,max_value_of_dtyp:27,maximum:27,mess:[],meta:[],method:24,metric:[2,5,11,13,14,16,17,24],metric_class:[13,17],metricmodel:24,might:[8,24],min:27,min_value_of_dtyp:27,mine:19,mini:13,minimum:27,mnist:[],mnist_load:[],mock:[],mode:[],model:[20,24],modul:[0,1],more:[5,24],muffin:7,multipl:[],name:[13,14,16,17,19,24],necessari:[],need:[],neg:[13,19],nn:7,non:[22,23,24],none:[8,15,20,22,23,24],normal:12,num_group:[12,18],num_work:[5,22,23],number:[12,18,24],obj:[5,7],obj_a:[5,7],obj_b:[5,7],object:[5,7,8,10,13,15,16,17,19,20,22,23,24,27],offset:5,onc:5,one:[5,7],onli:5,onlin:19,oper:12,option:[5,8,15,19,20,22,23,24],orang:[5,7],order:[],org:[12,19],origin:[4,5,24],original_param:5,other:[7,22,23],otherwis:[15,24],output:[5,12,18,24],output_path:[],overridden:8,overwrit:5,overwritten:5,packag:1,page:0,pair:[5,13,16],pairs_count:16,pairssimilaritydataload:5,pairwis:16,pairwise_loss:[2,11],pairwiseloss:[13,16],param:[5,13,17,19],paramet:[5,6,8,12,13,14,15,16,17,18,19,20,24,27],paramref:[],pass:[5,17,24,27],path:24,pattern:[],pdf:13,per:8,perform:[8,20],persist:[],person:7,pictur:7,pin_memori:5,pleas:[],posit:13,postiv:[],pre:[13,14,16,17],pre_collate_fn:[5,8],pre_encoder_col:8,predict:5,predict_dataload:[],predict_step:[],prefetch_factor:5,prepar:8,prepare_data:[],privat:[],process:[8,20,22,23,24],process_result:24,produc:24,product:[15,18],progress:24,properli:[],properti:[5,24],provid:24,pseudo:5,publi:13,purpos:[5,13,17,19],push:[12,19],pytorch:[24,27],pytorch_lightn:20,quaterion_model:24,queri:7,rais:[17,27],ram:24,random:5,raw:5,recommend:[],record:6,reduc:13,regular:18,reinforc:4,reinforce_typ:4,reload:[],reload_dataloaders_every_n_epoch:[],repeat:[],repres:7,requir:[4,5,8,22,23],reset:[],reset_cach:[],respect:5,restrict:4,result:[],retriev:[5,17,20],reus:[],root:[],routin:20,runtimeerror:17,s:[20,24],same:[],sampl:[5,13,20],sampler:5,save:[13,17,19,24],save_serv:24,scalar:19,scale:12,score:[5,7,13],search:0,second:[7,13],see:20,self:[],send:8,sequenc:[],sequenti:[],serializ:[8,13,17,19],serv:24,set:[22,23],setup:[],setup_dataload:24,shape:[12,14,15,16,18,19,24],should:[5,7,8,13,24],shoulder:20,shuffl:[],siamesedistancemetr:[13,14,15,16,17],similar:[5,7,16,17],similarity_data_load:[2,3],similarity_dataset:[2,3],similarity_loss:[2,11],similarity_sampl:[2,3],similaritydataload:[5,20,24],similaritygroupdataset:6,similaritygroupsampl:[5,6,7],similarityloss:[14,16,17,24],similaritypairsampl:[5,7],simpl:6,singl:7,situat:[],size:[13,14,16,18,22,23,24],size_averag:13,softmax:18,softmax_loss:[2,11],softmaxloss:18,some:24,sophist:24,sourc:[4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,22,23,24,26,27],special:5,specif:[5,24],specifi:[19,24],split:5,squar:[15,19],stage:[20,22,23,24,26],standard:6,state:24,store:24,str:[5,8,13,14,16,17,19,22,23,24,26],strategi:19,stuff:[],subgroup:[5,7,13,16],submodul:1,subpackag:1,subtyp:4,suitabl:5,sum:13,support:19,sure:15,t:[],t_co:[4,5],target:[5,24],task:6,temperatur:18,tensor:[5,8,10,12,13,14,15,16,17,18,19,22,23,24],test:[24,26],test_dataload:[],test_step:24,text:13,text_encod:24,than:5,thei:24,them:[],thi:[5,7,24],time:[],timeout:5,torch:[5,12,19,27],totensor:[],train:[1,2,5,8,12,13,14,16,17,18,19,20,26],train_collat:[2,3],train_dataload:[20,24],trainabl:24,trainable_model:[2,20,21],trainablemodel:[20,22,23,24],traincollat:8,trainer:[20,24],training_step:24,trainstag:[24,26],transform:5,triplet:19,triplet_loss:[2,11],tripletloss:19,tupl:[4,5,8,24],two:[13,14,16,17,20],type:[4,13,17,22,23,24,27],typeerror:27,understand:15,unexpect:15,union:[8,22,23,24,27],uniqu:[5,7],unknown:17,unless:[],updat:24,us:[5,13,14,16,17,19,22,23,24],util:[1,2,5],val_dataload:[20,24],valid:[20,24,26],validation_step:24,valu:[12,13,14,16,18,19,22,23,24,26,27],vector:24,vector_length:[12,14,16,18],version:5,wai:24,want:24,warn:[],well:24,what:24,when:[],where:[],whether:[],which:[5,6,7,13,24,27],whole:20,within:7,without:5,word:13,work:[5,12,15,18],worker:8,wrap:[],wrapper:6,x:15,y:15,yann:13,you:[15,24],yourself:[],zero:[14,16,18]},titles:["Welcome to Quaterion\u2019s documentation!","quaterion","quaterion package","quaterion.dataset package","quaterion.dataset.indexing_dataset module","quaterion.dataset.similarity_data_loader module","quaterion.dataset.similarity_dataset module","quaterion.dataset.similarity_samples module","quaterion.dataset.train_collater module","quaterion.eval package","quaterion.eval.base_metric module","quaterion.loss package","quaterion.loss.arcface_loss module","quaterion.loss.contrastive_loss module","quaterion.loss.group_loss module","quaterion.loss.metrics module","quaterion.loss.pairwise_loss module","quaterion.loss.similarity_loss module","quaterion.loss.softmax_loss module","quaterion.loss.triplet_loss module","quaterion.main module","quaterion.train package","quaterion.train.cache package","quaterion.train.cache.cache_config module","quaterion.train.trainable_model module","quaterion.utils package","quaterion.utils.enums module","quaterion.utils.utils module"],titleterms:{"enum":26,arcface_loss:12,base_metr:10,cach:[22,23],cache_config:23,cache_encod:[],cache_mixin:[],cache_model:[],cache_train_collat:[],content:[2,3,9,11,21,22,25],contrastive_loss:13,dataset:[3,4,5,6,7,8],document:0,eval:[9,10],group_loss:14,in_memory_cache_encod:[],indexing_dataset:4,indic:0,loss:[11,12,13,14,15,16,17,18,19],main:20,metric:15,modul:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],packag:[2,3,9,11,21,22,25],pairwise_loss:16,quaterion:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],s:0,similarity_data_load:5,similarity_dataset:6,similarity_loss:17,similarity_sampl:7,softmax_loss:18,submodul:[2,3,9,11,21,22,25],subpackag:[2,21],tabl:0,train:[21,22,23,24],train_collat:8,trainable_model:24,triplet_loss:19,util:[25,26,27],welcom:0}})