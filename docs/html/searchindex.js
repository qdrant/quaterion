Search.setIndex({docnames:["index","modules","quaterion","quaterion.dataset","quaterion.dataset.indexing_dataset","quaterion.dataset.similarity_data_loader","quaterion.dataset.similarity_dataset","quaterion.dataset.similarity_samples","quaterion.dataset.train_collater","quaterion.eval","quaterion.eval.base_metric","quaterion.loss","quaterion.loss.arcface_loss","quaterion.loss.contrastive_loss","quaterion.loss.group_loss","quaterion.loss.metrics","quaterion.loss.multiple_negatives_ranking_loss","quaterion.loss.pairwise_loss","quaterion.loss.similarity_loss","quaterion.loss.softmax_loss","quaterion.loss.triplet_loss","quaterion.main","quaterion.train","quaterion.train.cache","quaterion.train.cache.cache_config","quaterion.train.trainable_model","quaterion.utils","quaterion.utils.enums","quaterion.utils.utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["index.rst","modules.rst","quaterion.rst","quaterion.dataset.rst","quaterion.dataset.indexing_dataset.rst","quaterion.dataset.similarity_data_loader.rst","quaterion.dataset.similarity_dataset.rst","quaterion.dataset.similarity_samples.rst","quaterion.dataset.train_collater.rst","quaterion.eval.rst","quaterion.eval.base_metric.rst","quaterion.loss.rst","quaterion.loss.arcface_loss.rst","quaterion.loss.contrastive_loss.rst","quaterion.loss.group_loss.rst","quaterion.loss.metrics.rst","quaterion.loss.multiple_negatives_ranking_loss.rst","quaterion.loss.pairwise_loss.rst","quaterion.loss.similarity_loss.rst","quaterion.loss.softmax_loss.rst","quaterion.loss.triplet_loss.rst","quaterion.main.rst","quaterion.train.rst","quaterion.train.cache.rst","quaterion.train.cache.cache_config.rst","quaterion.train.trainable_model.rst","quaterion.utils.rst","quaterion.utils.enums.rst","quaterion.utils.utils.rst"],objects:{"":[[2,0,0,"-","quaterion"]],"quaterion.dataset":[[4,0,0,"-","indexing_dataset"],[5,0,0,"-","similarity_data_loader"],[6,0,0,"-","similarity_dataset"],[7,0,0,"-","similarity_samples"],[8,0,0,"-","train_collater"]],"quaterion.dataset.indexing_dataset":[[4,1,1,"","IndexingDataset"],[4,1,1,"","IndexingIterableDataset"]],"quaterion.dataset.indexing_dataset.IndexingIterableDataset":[[4,2,1,"","reinforce_type"]],"quaterion.dataset.similarity_data_loader":[[5,1,1,"","GroupSimilarityDataLoader"],[5,1,1,"","PairsSimilarityDataLoader"],[5,1,1,"","SimilarityDataLoader"]],"quaterion.dataset.similarity_data_loader.GroupSimilarityDataLoader":[[5,3,1,"","batch_size"],[5,2,1,"","collate_labels"],[5,3,1,"","dataset"],[5,3,1,"","drop_last"],[5,2,1,"","flatten_objects"],[5,3,1,"","num_workers"],[5,3,1,"","pin_memory"],[5,3,1,"","prefetch_factor"],[5,3,1,"","sampler"],[5,3,1,"","timeout"]],"quaterion.dataset.similarity_data_loader.PairsSimilarityDataLoader":[[5,3,1,"","batch_size"],[5,2,1,"","collate_labels"],[5,3,1,"","dataset"],[5,3,1,"","drop_last"],[5,2,1,"","flatten_objects"],[5,3,1,"","num_workers"],[5,3,1,"","pin_memory"],[5,3,1,"","prefetch_factor"],[5,3,1,"","sampler"],[5,3,1,"","timeout"]],"quaterion.dataset.similarity_data_loader.SimilarityDataLoader":[[5,3,1,"","batch_size"],[5,2,1,"","collate_labels"],[5,3,1,"","dataset"],[5,3,1,"","drop_last"],[5,2,1,"","flatten_objects"],[5,3,1,"","num_workers"],[5,4,1,"","original_params"],[5,3,1,"","pin_memory"],[5,2,1,"","pre_collate_fn"],[5,3,1,"","prefetch_factor"],[5,3,1,"","sampler"],[5,3,1,"","timeout"]],"quaterion.dataset.similarity_dataset":[[6,1,1,"","SimilarityGroupDataset"]],"quaterion.dataset.similarity_samples":[[7,1,1,"","SimilarityGroupSample"],[7,1,1,"","SimilarityPairSample"]],"quaterion.dataset.similarity_samples.SimilarityGroupSample":[[7,3,1,"","group"],[7,3,1,"","obj"]],"quaterion.dataset.similarity_samples.SimilarityPairSample":[[7,3,1,"","obj_a"],[7,3,1,"","obj_b"],[7,3,1,"","score"],[7,3,1,"","subgroup"]],"quaterion.dataset.train_collater":[[8,1,1,"","TrainCollater"]],"quaterion.dataset.train_collater.TrainCollater":[[8,2,1,"","pre_encoder_collate"]],"quaterion.eval":[[10,0,0,"-","base_metric"]],"quaterion.eval.base_metric":[[10,1,1,"","BaseMetric"]],"quaterion.eval.base_metric.BaseMetric":[[10,2,1,"","eval"]],"quaterion.loss":[[12,0,0,"-","arcface_loss"],[13,0,0,"-","contrastive_loss"],[14,0,0,"-","group_loss"],[15,0,0,"-","metrics"],[16,0,0,"-","multiple_negatives_ranking_loss"],[17,0,0,"-","pairwise_loss"],[18,0,0,"-","similarity_loss"],[19,0,0,"-","softmax_loss"],[20,0,0,"-","triplet_loss"]],"quaterion.loss.arcface_loss":[[12,1,1,"","ArcFaceLoss"],[12,5,1,"","l2_norm"]],"quaterion.loss.arcface_loss.ArcFaceLoss":[[12,2,1,"","forward"],[12,3,1,"","training"]],"quaterion.loss.contrastive_loss":[[13,1,1,"","ContrastiveLoss"]],"quaterion.loss.contrastive_loss.ContrastiveLoss":[[13,2,1,"","forward"],[13,2,1,"","get_config_dict"],[13,2,1,"","metric_class"],[13,3,1,"","training"]],"quaterion.loss.group_loss":[[14,1,1,"","GroupLoss"]],"quaterion.loss.group_loss.GroupLoss":[[14,2,1,"","forward"],[14,3,1,"","training"]],"quaterion.loss.metrics":[[15,1,1,"","SiameseDistanceMetric"]],"quaterion.loss.metrics.SiameseDistanceMetric":[[15,2,1,"","cosine_distance"],[15,2,1,"","dot_product_distance"],[15,2,1,"","euclidean"],[15,2,1,"","manhattan"]],"quaterion.loss.multiple_negatives_ranking_loss":[[16,1,1,"","MultipleNegativesRankingLoss"]],"quaterion.loss.multiple_negatives_ranking_loss.MultipleNegativesRankingLoss":[[16,2,1,"","forward"],[16,2,1,"","get_config_dict"],[16,2,1,"","metric_class"],[16,3,1,"","training"]],"quaterion.loss.pairwise_loss":[[17,1,1,"","PairwiseLoss"]],"quaterion.loss.pairwise_loss.PairwiseLoss":[[17,2,1,"","forward"],[17,3,1,"","training"]],"quaterion.loss.similarity_loss":[[18,1,1,"","SimilarityLoss"]],"quaterion.loss.similarity_loss.SimilarityLoss":[[18,2,1,"","get_config_dict"],[18,2,1,"","get_distance_function"],[18,2,1,"","metric_class"],[18,3,1,"","training"]],"quaterion.loss.softmax_loss":[[19,1,1,"","SoftmaxLoss"]],"quaterion.loss.softmax_loss.SoftmaxLoss":[[19,2,1,"","forward"],[19,3,1,"","training"]],"quaterion.loss.triplet_loss":[[20,1,1,"","TripletLoss"]],"quaterion.loss.triplet_loss.TripletLoss":[[20,2,1,"","forward"],[20,2,1,"","get_config_dict"],[20,3,1,"","training"]],"quaterion.main":[[21,1,1,"","Quaterion"]],"quaterion.main.Quaterion":[[21,2,1,"","fit"]],"quaterion.train":[[23,0,0,"-","cache"],[25,0,0,"-","trainable_model"]],"quaterion.train.cache":[[23,1,1,"","CacheConfig"],[23,1,1,"","CacheType"],[24,0,0,"-","cache_config"]],"quaterion.train.cache.CacheConfig":[[23,3,1,"","batch_size"],[23,3,1,"","cache_type"],[23,3,1,"","key_extractors"],[23,3,1,"","mapping"],[23,3,1,"","num_workers"]],"quaterion.train.cache.CacheType":[[23,3,1,"","AUTO"],[23,3,1,"","CPU"],[23,3,1,"","GPU"]],"quaterion.train.cache.cache_config":[[24,1,1,"","CacheConfig"],[24,1,1,"","CacheType"]],"quaterion.train.cache.cache_config.CacheConfig":[[24,3,1,"","batch_size"],[24,3,1,"","cache_type"],[24,3,1,"","key_extractors"],[24,3,1,"","mapping"],[24,3,1,"","num_workers"]],"quaterion.train.cache.cache_config.CacheType":[[24,3,1,"","AUTO"],[24,3,1,"","CPU"],[24,3,1,"","GPU"]],"quaterion.train.trainable_model":[[25,1,1,"","TrainableModel"]],"quaterion.train.trainable_model.TrainableModel":[[25,2,1,"","cache"],[25,2,1,"","configure_caches"],[25,2,1,"","configure_encoders"],[25,2,1,"","configure_head"],[25,2,1,"","configure_loss"],[25,4,1,"","loss"],[25,4,1,"","model"],[25,2,1,"","process_results"],[25,2,1,"","save_servable"],[25,2,1,"","setup_dataloader"],[25,2,1,"","test_step"],[25,3,1,"","training"],[25,2,1,"","training_step"],[25,2,1,"","unwrap_cache"],[25,2,1,"","validation_step"]],"quaterion.utils":[[27,0,0,"-","enums"],[28,0,0,"-","utils"]],"quaterion.utils.enums":[[27,1,1,"","TrainStage"]],"quaterion.utils.enums.TrainStage":[[27,3,1,"","TEST"],[27,3,1,"","TRAIN"],[27,3,1,"","VALIDATION"]],"quaterion.utils.utils":[[28,5,1,"","info_value_of_dtype"],[28,5,1,"","max_value_of_dtype"],[28,5,1,"","min_value_of_dtype"]],quaterion:[[3,0,0,"-","dataset"],[9,0,0,"-","eval"],[11,0,0,"-","loss"],[21,0,0,"-","main"],[22,0,0,"-","train"],[26,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","property","Python property"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:property","5":"py:function"},terms:{"0":[5,7,12,13,16,19,20],"00652":16,"03832":20,"05":19,"06":13,"07698":12,"1":[5,7,13,15,16],"10":7,"11":7,"1503":20,"1705":16,"1801":12,"1st_pair_1st_obj":5,"1st_pair_2nd_obj":5,"2":[5,7,17],"20":16,"209":7,"2nd_pair_1st_obj":5,"2nd_pair_2nd_obj":5,"3":[5,7],"32":[23,24],"4":7,"5":[12,20],"555":7,"64":12,"7":7,"8":7,"9":7,"case":7,"class":[4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,21,23,24,25,27],"default":[8,19,20,25],"do":25,"enum":[2,23,24,26],"float":[5,7,12,13,16,19,20,28],"function":[5,8,13,14,15,16,17,18,20,23,24,25],"int":[5,7,8,12,19,23,24,25,28],"new":5,"return":[5,6,12,13,14,15,16,17,18,19,20,25,28],"static":15,"true":[13,15,16,20],A:21,And:4,If:[13,15,16],In:7,It:[16,19,20,23,24],One:20,The:[13,14,17,18],Then:16,__init__:5,_s:5,ab:[12,20],about:28,accept:16,account:16,actual:5,ad:25,addit:[5,12,13,16,25],addition:5,affect:[23,24],aggreg:8,all:[7,8,15,20,25],allow:28,also:16,among:8,an:[16,19,25],anchor:16,angular:12,ani:[4,5,7,8,13,16,18,20,25],anoth:7,answer:16,apart:[12,20],appl:5,appli:[5,12,25],ar:[5,13,16],arcface_loss:[2,11],arcfaceloss:12,arg:25,argument:[5,13,16,25],arxiv:[12,16,20],assembl:21,assign:[5,15,25],associ:[5,12,14,19,20,25],associat:5,attribut:18,auto:[23,24,25],automat:[5,16,25],avail:[13,16,18,23,24],averag:13,bach:5,bar:25,base:[4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,21,23,24,25,27],base_metr:[2,9],basemetr:10,batch:[5,8,13,16,17,20,23,24,25],batch_idx:25,batch_siz:[5,12,14,15,17,19,20,23,24,25],befor:[5,25],between:[13,14,15,16,17,18,25],bool:[5,12,13,14,15,16,17,18,19,20,25,28],cach:[2,21,22,25],cachabl:25,cache_config:[22,23],cache_typ:[23,24,25],cacheabl:[23,24],cacheconfig:[23,24,25],cachedataload:[23,24],cachemixin:25,cachetyp:[23,24,25],calcul:[15,16,18,20,25],call:[13,16],callabl:18,can:[13,14,17,18],candi:7,cannot:13,checkpoint:25,cheesecak:7,chopra:13,chosen:15,cl:18,classif:6,classmethod:[5,13,16,18,21],closer:7,collat:[5,8,25],collate_fn:5,collate_label:5,com:13,common:15,compat:6,comput:[12,13,15,16,17,19,25],config:[13,16,18,20],configur:25,configure_cach:[23,24,25],configure_encod:25,configure_head:25,configure_loss:25,consum:5,contain:[13,14,16,17,18],content:1,contrast:13,contrastive_loss:[2,11],contrastiveloss:13,convert:[5,6],correct:16,correspond:[13,16],cosin:[15,16],cosine_dist:[13,14,15,17,18],cpu:[23,24,25],cross:[12,16,19],cuda:[23,24],current:[13,16,18],data:[5,6,7,21,25,28],dataload:[5,21,25],datapip:4,dataset:[1,2,12,19,25],debug:5,default_encoder_kei:25,defin:[5,12,13,14,17,18,20,25],describ:16,design:19,devic:[23,24],dict:[5,13,16,18,20,23,24,25],differ:7,dim:12,dimens:[12,19],directli:5,disambigu:25,displai:25,distanc:[13,14,15,17,18,20],distance_metric_nam:[13,14,17,18,20],distinguish:13,divid:19,doe:[23,24,28],dot:[15,19],dot_product:16,dot_product_dist:15,drop_last:5,dtype:28,dummi:5,duplic:16,dure:[21,23,24,25],dwarf:21,e:[16,25],each:[5,7,25],effect:15,either:[13,16],elon_musk_1:7,elon_musk_2:7,elon_musk_3:7,els:[23,24],embed:[12,13,14,16,17,18,19,20,25],embedding_dim:20,embedding_s:[12,19,25],encod:[5,8,12,19,23,24,25],encoder_col:8,encoder_nam:[8,23,24],encoderhead:25,encor:5,entir:15,entropi:[12,16,19],enumer:5,especi:15,estim:5,euclidean:[15,20],eval:[1,2],evalu:25,exampl:[5,7,13,20,25],exdb:13,expect:13,expected_typ:4,extractor:[23,24],face:7,factori:[23,24],fals:[15,16],farther:21,featir:5,featur:[5,8],file_nam:7,fill:25,finfo:28,first:[7,13,16],fit:21,flat:15,flatten_object:5,form:16,format:6,forward:[12,13,14,16,17,19,20],from:[5,7,12,16,18,19,25],function_nam:18,further:[7,13],g:25,gener:[5,13,16],get:25,get_collate_fn:5,get_config_dict:[13,16,18,20],get_distance_funct:18,giant:21,given:[16,28],gpu:[23,24,25],great:16,group:[5,7,12,14,19,20],group_id:7,group_loss:[2,11],grouploss:[12,14,19,20],groupsimilaritydataload:[5,6],ha:[13,16,18,20],hadsel:13,half:[13,16],handl:[21,27],hard:20,hash:25,hash_id:5,hashabl:[23,24],have:[7,13,15],head:25,hint:4,how:15,http:[12,13,16,20],i:16,id:[5,7,8,12,17,19,25],ignor:16,iinfo:28,image_encod:25,implement:[8,16,19,20],increas:13,independ:25,index:[0,25],indexing_dataset:[2,3],indexingdataset:4,indexingiterabledataset:4,indic:[13,16],individu:[5,25],info:28,info_value_of_dtyp:28,inform:[8,13],initi:[5,25],input:[5,12,13],input_embedding_s:25,instanc:[4,18,21,25],integ:25,intern:21,item:[5,25],iterabledataset:4,itself:15,jpg:7,json:[13,16,18,20],kei:[13,16,23,24,25],key_extractor:[23,24,25],keyextractortyp:[23,24],keyword:25,kind:25,known:17,kwarg:[5,13,16,25],l2:12,l2_norm:12,label:[5,6,13,16,17,20],labels_batch:5,layer:25,learn:7,least:13,lecun:13,lemon:[5,7],leonard_nimoy_1:7,leonard_nimoy_2:7,lightn:25,lightningmodul:25,likelihood:16,lime:7,list:[5,8,17,25],load:[13,16,18,20,25],loader:[21,25],log:16,logger:25,logit:19,longtensor:[13,14,16,19,20],loss:[1,2,5,25],macaroon:7,mai:15,main:[1,2],make:[12,15,16],manhattan:15,map:[23,24,25],margin:[12,13,20],match:7,matrix:[15,16],max:28,max_value_of_dtyp:28,maximum:28,method:25,metric:[2,5,11,13,14,16,17,18,25],metric_class:[13,16,18],metricmodel:25,might:[8,25],min:28,min_value_of_dtyp:28,mine:20,mini:13,minim:16,minimum:28,modal:16,model:[21,25],modul:[0,1],more:[5,25],muffin:7,multipl:16,multiple_negatives_ranking_loss:[2,11],multiplenegativesrankingloss:16,multipli:16,must:16,name:[13,14,16,17,18,20,25],neg:[13,16,20],nn:7,non:[16,23,24,25],none:[4,8,15,21,23,24,25],normal:[12,16],num_group:[12,19],num_work:[5,23,24],number:[12,19,25],obj:[5,7],obj_a:[5,7],obj_b:[5,7],object:[5,7,8,10,13,15,16,17,18,20,21,23,24,25,28],offset:5,onc:5,one:[5,7],onli:[5,16],onlin:20,oper:12,optim:16,option:[5,8,15,20,21,23,24,25],orang:[5,7],org:[12,16,20],origin:[4,5,25],original_param:5,other:[7,23,24],otherwis:[15,25],output:[5,12,19,25],overridden:8,overwrit:5,overwritten:5,packag:1,page:0,pair:[5,13,16,17],pairs_count:17,pairssimilaritydataload:5,pairwis:17,pairwise_loss:[2,11],pairwiseloss:[13,16,17],param:[5,13,16,18,20],paramet:[5,6,8,12,13,14,15,16,17,18,19,20,21,25,28],pass:[5,18,25,28],path:25,pdf:[13,16],per:8,perform:[8,21],person:7,pictur:7,pin_memori:5,posit:[13,16],pre:[13,14,17,18],pre_collate_fn:[5,8],pre_encoder_col:8,predict:5,prefetch_factor:5,prepar:8,process:[8,21,23,24,25],process_result:25,produc:25,product:[15,19],progress:25,properti:[5,25],provid:25,pseudo:5,publi:13,purpos:[5,13,16,18,20],push:[12,20],pytorch:[25,28],pytorch_lightn:21,quaterion_model:25,queri:7,question:16,rais:[18,28],ram:25,random:5,rank:16,raw:5,record:6,reduc:13,regular:19,reinforc:4,reinforce_typ:4,repres:7,requir:[4,5,8,23,24],respect:5,restor:25,restrict:4,retriev:[5,16,18,21],routin:21,runtimeerror:18,s:[21,25],sampl:[5,13,16,21],sampler:5,save:[13,16,18,20,25],save_serv:25,scalar:[16,20],scale:[12,16],score:[5,7,13,16],search:0,second:[7,13,16],see:21,seed:4,send:8,sentenc:16,serializ:[8,13,16,18,20],serv:25,set:[23,24],setup_dataload:25,shape:[12,14,15,17,19,20,25],should:[5,7,8,13,25],shoulder:21,siamesedistancemetr:[13,14,15,17,18],similar:[5,7,16,17,18],similarity_data_load:[2,3],similarity_dataset:[2,3],similarity_loss:[2,11],similarity_metric_nam:16,similarity_sampl:[2,3],similaritydataload:[5,21,25],similaritygroupdataset:6,similaritygroupsampl:[5,6,7],similarityloss:[14,17,18,25],similaritypairsampl:[5,7],simpl:6,singl:7,size:[13,14,17,19,23,24,25],size_averag:13,softmax:[16,19],softmax_loss:[2,11],softmaxloss:19,some:25,sophist:25,sourc:[4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,21,23,24,25,27,28],special:5,specif:[5,25],specifi:[20,25],split:5,squar:[15,20],stage:[21,23,24,25,27],standard:6,state:25,store:25,str:[5,8,13,14,16,17,18,20,23,24,25,27],strategi:20,subgroup:[5,7,13,16,17],submodul:1,subpackag:1,subtyp:4,suitabl:5,sum:13,support:20,sure:15,symmetr:16,t_co:[4,5],target:[5,25],task:[6,16],temperatur:19,tensor:[5,10,12,13,14,15,16,17,18,19,20,23,24,25],test:[25,27],test_step:25,text:13,text_encod:25,than:5,thei:25,them:16,thi:[5,7,16,25],timeout:5,torch:[5,12,20,28],train:[1,2,5,8,12,13,14,16,17,18,19,20,21,27],train_collat:[2,3],train_dataload:[21,25],trainabl:25,trainable_model:[2,21,22],trainablemodel:[21,23,24,25],traincollat:8,trainer:[21,25],training_step:25,trainstag:[25,27],transform:5,triplet:20,triplet_loss:[2,11],tripletloss:20,tupl:[4,5,25],two:[13,14,17,18,21],type:[4,13,16,18,23,24,25,28],typeerror:28,understand:15,unexpect:15,union:[23,24,25,28],uniqu:[5,7],unknown:18,unwrap_cach:25,updat:25,us:[5,13,14,16,17,18,20,23,24,25],util:[1,2,5],val_dataload:[21,25],valid:[21,25,27],validation_step:25,valu:[12,13,14,16,17,19,20,23,24,25,27,28],vector:25,vector_length:[12,14,17,19],version:5,wai:25,want:25,well:25,what:25,when:16,which:[5,6,7,13,25,28],whole:21,within:7,without:5,word:[13,16],work:[5,12,15,16,19],worker:8,wrapper:6,x:15,y:15,yann:13,you:[15,25],zero:[14,17,19]},titles:["Welcome to Quaterion\u2019s documentation!","quaterion","quaterion package","quaterion.dataset package","quaterion.dataset.indexing_dataset module","quaterion.dataset.similarity_data_loader module","quaterion.dataset.similarity_dataset module","quaterion.dataset.similarity_samples module","quaterion.dataset.train_collater module","quaterion.eval package","quaterion.eval.base_metric module","quaterion.loss package","quaterion.loss.arcface_loss module","quaterion.loss.contrastive_loss module","quaterion.loss.group_loss module","quaterion.loss.metrics module","quaterion.loss.multiple_negatives_ranking_loss module","quaterion.loss.pairwise_loss module","quaterion.loss.similarity_loss module","quaterion.loss.softmax_loss module","quaterion.loss.triplet_loss module","quaterion.main module","quaterion.train package","quaterion.train.cache package","quaterion.train.cache.cache_config module","quaterion.train.trainable_model module","quaterion.utils package","quaterion.utils.enums module","quaterion.utils.utils module"],titleterms:{"enum":27,arcface_loss:12,base_metr:10,cach:[23,24],cache_config:24,content:[2,3,9,11,22,23,26],contrastive_loss:13,dataset:[3,4,5,6,7,8],document:0,eval:[9,10],group_loss:14,indexing_dataset:4,indic:0,loss:[11,12,13,14,15,16,17,18,19,20],main:21,metric:15,modul:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],multiple_negatives_ranking_loss:16,packag:[2,3,9,11,22,23,26],pairwise_loss:17,quaterion:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],s:0,similarity_data_load:5,similarity_dataset:6,similarity_loss:18,similarity_sampl:7,softmax_loss:19,submodul:[2,3,9,11,22,23,26],subpackag:[2,22],tabl:0,train:[22,23,24,25],train_collat:8,trainable_model:25,triplet_loss:20,util:[26,27,28],welcom:0}})