Search.setIndex({docnames:["index","modules","quaterion","quaterion.dataset","quaterion.dataset.indexing_dataset","quaterion.dataset.similarity_data_loader","quaterion.dataset.similarity_dataset","quaterion.dataset.similarity_samples","quaterion.dataset.train_collater","quaterion.eval","quaterion.eval.base_metric","quaterion.loss","quaterion.loss.arcface_loss","quaterion.loss.contrastive_loss","quaterion.loss.group_loss","quaterion.loss.metrics","quaterion.loss.multiple_negatives_ranking_loss","quaterion.loss.online_contrastive_loss","quaterion.loss.pairwise_loss","quaterion.loss.similarity_loss","quaterion.loss.softmax_loss","quaterion.loss.triplet_loss","quaterion.main","quaterion.train","quaterion.train.cache","quaterion.train.cache.cache_config","quaterion.train.trainable_model","quaterion.utils","quaterion.utils.enums","quaterion.utils.utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["index.rst","modules.rst","quaterion.rst","quaterion.dataset.rst","quaterion.dataset.indexing_dataset.rst","quaterion.dataset.similarity_data_loader.rst","quaterion.dataset.similarity_dataset.rst","quaterion.dataset.similarity_samples.rst","quaterion.dataset.train_collater.rst","quaterion.eval.rst","quaterion.eval.base_metric.rst","quaterion.loss.rst","quaterion.loss.arcface_loss.rst","quaterion.loss.contrastive_loss.rst","quaterion.loss.group_loss.rst","quaterion.loss.metrics.rst","quaterion.loss.multiple_negatives_ranking_loss.rst","quaterion.loss.online_contrastive_loss.rst","quaterion.loss.pairwise_loss.rst","quaterion.loss.similarity_loss.rst","quaterion.loss.softmax_loss.rst","quaterion.loss.triplet_loss.rst","quaterion.main.rst","quaterion.train.rst","quaterion.train.cache.rst","quaterion.train.cache.cache_config.rst","quaterion.train.trainable_model.rst","quaterion.utils.rst","quaterion.utils.enums.rst","quaterion.utils.utils.rst"],objects:{"":[[2,0,0,"-","quaterion"]],"quaterion.dataset":[[4,0,0,"-","indexing_dataset"],[5,0,0,"-","similarity_data_loader"],[6,0,0,"-","similarity_dataset"],[7,0,0,"-","similarity_samples"],[8,0,0,"-","train_collater"]],"quaterion.dataset.indexing_dataset":[[4,1,1,"","IndexingDataset"],[4,1,1,"","IndexingIterableDataset"]],"quaterion.dataset.indexing_dataset.IndexingIterableDataset":[[4,2,1,"","reinforce_type"]],"quaterion.dataset.similarity_data_loader":[[5,1,1,"","GroupSimilarityDataLoader"],[5,1,1,"","PairsSimilarityDataLoader"],[5,1,1,"","SimilarityDataLoader"]],"quaterion.dataset.similarity_data_loader.GroupSimilarityDataLoader":[[5,3,1,"","batch_size"],[5,2,1,"","collate_labels"],[5,3,1,"","dataset"],[5,3,1,"","drop_last"],[5,2,1,"","flatten_objects"],[5,3,1,"","num_workers"],[5,3,1,"","pin_memory"],[5,3,1,"","prefetch_factor"],[5,3,1,"","sampler"],[5,3,1,"","timeout"]],"quaterion.dataset.similarity_data_loader.PairsSimilarityDataLoader":[[5,3,1,"","batch_size"],[5,2,1,"","collate_labels"],[5,3,1,"","dataset"],[5,3,1,"","drop_last"],[5,2,1,"","flatten_objects"],[5,3,1,"","num_workers"],[5,3,1,"","pin_memory"],[5,3,1,"","prefetch_factor"],[5,3,1,"","sampler"],[5,3,1,"","timeout"]],"quaterion.dataset.similarity_data_loader.SimilarityDataLoader":[[5,3,1,"","batch_size"],[5,2,1,"","collate_labels"],[5,3,1,"","dataset"],[5,3,1,"","drop_last"],[5,2,1,"","flatten_objects"],[5,3,1,"","num_workers"],[5,4,1,"","original_params"],[5,3,1,"","pin_memory"],[5,2,1,"","pre_collate_fn"],[5,3,1,"","prefetch_factor"],[5,3,1,"","sampler"],[5,3,1,"","timeout"]],"quaterion.dataset.similarity_dataset":[[6,1,1,"","SimilarityGroupDataset"]],"quaterion.dataset.similarity_samples":[[7,1,1,"","SimilarityGroupSample"],[7,1,1,"","SimilarityPairSample"]],"quaterion.dataset.similarity_samples.SimilarityGroupSample":[[7,3,1,"","group"],[7,3,1,"","obj"]],"quaterion.dataset.similarity_samples.SimilarityPairSample":[[7,3,1,"","obj_a"],[7,3,1,"","obj_b"],[7,3,1,"","score"],[7,3,1,"","subgroup"]],"quaterion.dataset.train_collater":[[8,1,1,"","TrainCollater"]],"quaterion.dataset.train_collater.TrainCollater":[[8,2,1,"","pre_encoder_collate"]],"quaterion.eval":[[10,0,0,"-","base_metric"]],"quaterion.eval.base_metric":[[10,1,1,"","BaseMetric"]],"quaterion.eval.base_metric.BaseMetric":[[10,2,1,"","eval"]],"quaterion.loss":[[12,0,0,"-","arcface_loss"],[13,0,0,"-","contrastive_loss"],[14,0,0,"-","group_loss"],[15,0,0,"-","metrics"],[16,0,0,"-","multiple_negatives_ranking_loss"],[17,0,0,"-","online_contrastive_loss"],[18,0,0,"-","pairwise_loss"],[19,0,0,"-","similarity_loss"],[20,0,0,"-","softmax_loss"],[21,0,0,"-","triplet_loss"]],"quaterion.loss.arcface_loss":[[12,1,1,"","ArcFaceLoss"],[12,5,1,"","l2_norm"]],"quaterion.loss.arcface_loss.ArcFaceLoss":[[12,2,1,"","forward"],[12,3,1,"","training"]],"quaterion.loss.contrastive_loss":[[13,1,1,"","ContrastiveLoss"]],"quaterion.loss.contrastive_loss.ContrastiveLoss":[[13,2,1,"","forward"],[13,2,1,"","get_config_dict"],[13,2,1,"","metric_class"],[13,3,1,"","training"]],"quaterion.loss.group_loss":[[14,1,1,"","GroupLoss"]],"quaterion.loss.group_loss.GroupLoss":[[14,2,1,"","forward"],[14,3,1,"","training"]],"quaterion.loss.metrics":[[15,1,1,"","SiameseDistanceMetric"]],"quaterion.loss.metrics.SiameseDistanceMetric":[[15,2,1,"","cosine_distance"],[15,2,1,"","dot_product_distance"],[15,2,1,"","euclidean"],[15,2,1,"","manhattan"]],"quaterion.loss.multiple_negatives_ranking_loss":[[16,1,1,"","MultipleNegativesRankingLoss"]],"quaterion.loss.multiple_negatives_ranking_loss.MultipleNegativesRankingLoss":[[16,2,1,"","forward"],[16,2,1,"","get_config_dict"],[16,2,1,"","metric_class"],[16,3,1,"","training"]],"quaterion.loss.online_contrastive_loss":[[17,1,1,"","OnlineContrastiveLoss"]],"quaterion.loss.online_contrastive_loss.OnlineContrastiveLoss":[[17,2,1,"","forward"],[17,2,1,"","get_config_dict"],[17,3,1,"","training"]],"quaterion.loss.pairwise_loss":[[18,1,1,"","PairwiseLoss"]],"quaterion.loss.pairwise_loss.PairwiseLoss":[[18,2,1,"","forward"],[18,3,1,"","training"]],"quaterion.loss.similarity_loss":[[19,1,1,"","SimilarityLoss"]],"quaterion.loss.similarity_loss.SimilarityLoss":[[19,2,1,"","get_config_dict"],[19,2,1,"","get_distance_function"],[19,2,1,"","metric_class"],[19,3,1,"","training"]],"quaterion.loss.softmax_loss":[[20,1,1,"","SoftmaxLoss"]],"quaterion.loss.softmax_loss.SoftmaxLoss":[[20,2,1,"","forward"],[20,3,1,"","training"]],"quaterion.loss.triplet_loss":[[21,1,1,"","TripletLoss"]],"quaterion.loss.triplet_loss.TripletLoss":[[21,2,1,"","forward"],[21,2,1,"","get_config_dict"],[21,3,1,"","training"]],"quaterion.main":[[22,1,1,"","Quaterion"]],"quaterion.main.Quaterion":[[22,2,1,"","fit"]],"quaterion.train":[[24,0,0,"-","cache"],[26,0,0,"-","trainable_model"]],"quaterion.train.cache":[[24,1,1,"","CacheConfig"],[24,1,1,"","CacheType"],[25,0,0,"-","cache_config"]],"quaterion.train.cache.CacheConfig":[[24,3,1,"","batch_size"],[24,3,1,"","cache_type"],[24,3,1,"","key_extractors"],[24,3,1,"","mapping"],[24,3,1,"","num_workers"]],"quaterion.train.cache.CacheType":[[24,3,1,"","AUTO"],[24,3,1,"","CPU"],[24,3,1,"","GPU"]],"quaterion.train.cache.cache_config":[[25,1,1,"","CacheConfig"],[25,1,1,"","CacheType"]],"quaterion.train.cache.cache_config.CacheConfig":[[25,3,1,"","batch_size"],[25,3,1,"","cache_type"],[25,3,1,"","key_extractors"],[25,3,1,"","mapping"],[25,3,1,"","num_workers"]],"quaterion.train.cache.cache_config.CacheType":[[25,3,1,"","AUTO"],[25,3,1,"","CPU"],[25,3,1,"","GPU"]],"quaterion.train.trainable_model":[[26,1,1,"","TrainableModel"]],"quaterion.train.trainable_model.TrainableModel":[[26,2,1,"","cache"],[26,2,1,"","configure_caches"],[26,2,1,"","configure_encoders"],[26,2,1,"","configure_head"],[26,2,1,"","configure_loss"],[26,4,1,"","loss"],[26,4,1,"","model"],[26,2,1,"","process_results"],[26,2,1,"","save_servable"],[26,2,1,"","setup_dataloader"],[26,2,1,"","test_step"],[26,3,1,"","training"],[26,2,1,"","training_step"],[26,2,1,"","unwrap_cache"],[26,2,1,"","validation_step"]],"quaterion.utils":[[28,0,0,"-","enums"],[29,0,0,"-","utils"]],"quaterion.utils.enums":[[28,1,1,"","TrainStage"]],"quaterion.utils.enums.TrainStage":[[28,3,1,"","TEST"],[28,3,1,"","TRAIN"],[28,3,1,"","VALIDATION"]],"quaterion.utils.utils":[[29,5,1,"","get_anchor_negative_mask"],[29,5,1,"","get_anchor_positive_mask"],[29,5,1,"","get_triplet_mask"],[29,5,1,"","info_value_of_dtype"],[29,5,1,"","max_value_of_dtype"],[29,5,1,"","min_value_of_dtype"]],quaterion:[[3,0,0,"-","dataset"],[9,0,0,"-","eval"],[11,0,0,"-","loss"],[22,0,0,"-","main"],[23,0,0,"-","train"],[27,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","property","Python property"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:property","5":"py:function"},terms:{"0":[5,7,12,13,16,17,20,21],"00652":16,"03832":21,"05":20,"06":[13,17],"07698":12,"1":[5,7,13,15,16],"10":7,"11":7,"1503":21,"1705":16,"1801":12,"1st_pair_1st_obj":5,"1st_pair_2nd_obj":5,"2":[5,7,18],"20":16,"209":7,"2d":29,"2nd_pair_1st_obj":5,"2nd_pair_2nd_obj":5,"3":[5,7,29],"32":[24,25],"3d":29,"4":7,"5":[12,17,21],"555":7,"64":12,"7":7,"8":7,"9":7,"case":7,"class":[4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,21,22,24,25,26,28],"default":[8,17,20,21,26],"do":26,"enum":[2,24,25,27],"float":[5,7,12,13,16,17,20,21,29],"function":[5,8,13,14,15,16,17,18,19,21,24,25,26,29],"int":[5,7,8,12,20,24,25,26,29],"new":5,"return":[5,6,12,13,14,15,16,17,18,19,20,21,26,29],"static":15,"true":[13,15,16,17,21],A:22,And:4,For:16,If:[13,15,16],In:7,It:[16,20,21,24,25],One:[17,21],The:[13,14,18,19],Then:16,__init__:5,_s:5,ab:[12,21],about:[16,29],abov:29,accept:16,account:16,actual:[5,29],ad:26,addit:[5,12,13,16,26],addition:5,affect:[24,25],aggreg:8,all:[7,8,15,17,21,26,29],allow:29,also:16,among:8,an:[16,20,26],anchor:[16,29],angular:12,ani:[4,5,7,8,13,16,17,19,21,26],anoth:7,answer:16,apart:[12,17,21],appl:5,appli:[5,12,26],ar:[5,13,16,29],arcface_loss:[2,11],arcfaceloss:12,arg:26,argument:[5,13,16,26],arxiv:[12,16,21],assembl:22,assign:[5,15,26],associ:[5,12,14,17,20,21,26,29],associat:5,assum:16,attribut:19,auto:[24,25,26],automat:[5,16,26],avail:[13,16,19,24,25],averag:13,bach:5,bar:26,base:[4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,21,22,24,25,26,28,29],base_metr:[2,9],basemetr:10,batch:[5,8,13,16,17,18,21,24,25,26,29],batch_idx:26,batch_siz:[5,12,14,15,17,18,20,21,24,25,26,29],befor:[5,26],between:[13,14,15,16,18,19,26],bool:[5,12,13,14,15,16,17,18,19,20,21,26,29],cach:[2,22,23,26],cachabl:26,cache_config:[23,24],cache_typ:[24,25,26],cacheabl:[24,25],cacheconfig:[24,25,26],cachedataload:[24,25],cachemixin:26,cachetyp:[24,25,26],calcul:[15,16,17,19,21,26,29],call:[13,16],callabl:19,can:[13,14,18,19,29],candi:7,cannot:13,checkpoint:26,cheesecak:7,chopra:[13,17],chosen:15,cl:19,classif:6,classmethod:[5,13,16,19,22],closer:7,collat:[5,8,26],collate_fn:5,collate_label:5,com:[13,17],common:15,compat:6,comput:[12,13,15,16,18,20,26],config:[13,16,17,19,21],configur:26,configure_cach:[24,25,26],configure_encod:26,configure_head:26,configure_loss:26,consum:5,contain:[13,14,16,18,19],content:1,contrast:[13,17],contrastive_loss:[2,11],contrastiveloss:[13,17],convert:[5,6],correct:16,correspond:[13,16],cosin:[15,16],cosine_dist:[13,14,15,17,18,19],cpu:[24,25,26],creat:29,criteria:29,cross:[12,16,20],cube:29,cuda:[24,25],current:[13,16,19],data:[5,6,7,22,26,29],dataload:[5,22,26],datapip:4,dataset:[1,2,12,20,26],debug:5,default_encoder_kei:26,defin:[5,12,13,14,17,18,19,21,26],describ:16,design:20,devic:[24,25],dict:[5,13,16,17,19,21,24,25,26],differ:7,dim:12,dimens:[12,20],directli:5,disambigu:26,displai:26,distanc:[13,14,15,17,18,19,21],distance_metric_nam:[13,14,17,18,19,21],distinct:29,distinguish:13,divid:20,doe:[24,25,29],don:16,dot:[15,20],dot_product:16,dot_product_dist:15,drop_last:5,dtype:29,dummi:5,duplic:16,dure:[22,24,25,26],dwarf:22,e:[16,26,29],each:[5,7,16,26],effect:15,either:[13,16],elon_musk_1:7,elon_musk_2:7,elon_musk_3:7,els:[24,25],embed:[12,13,14,16,17,18,19,20,21,26,29],embedding_dim:[17,21],embedding_s:[12,20,26],encod:[5,8,12,20,24,25,26],encoder_col:8,encoder_nam:[8,24,25],encoderhead:26,encor:5,entir:15,entropi:[12,16,20],enumer:5,especi:15,estim:5,euclidean:[15,17,21],eval:[1,2],evalu:26,exampl:[5,7,13,16,17,21,26],exdb:[13,17],expect:13,expected_typ:4,extractor:[24,25],face:7,factori:[24,25],fals:[15,16],farther:22,featir:5,featur:[5,8],file_nam:7,fill:26,finfo:29,first:[7,13,16],fit:22,flat:15,flatten_object:5,form:[16,29],format:6,forward:[12,13,14,16,17,18,20,21],from:[5,7,12,16,19,20,26],function_nam:19,further:[7,13],g:[16,26],gener:[5,13,16],get:26,get_anchor_negative_mask:29,get_anchor_positive_mask:29,get_collate_fn:5,get_config_dict:[13,16,17,19,21],get_distance_funct:19,get_triplet_mask:29,giant:22,given:[16,29],gpu:[24,25,26],great:16,group:[5,7,12,14,17,20,21],group_id:7,group_loss:[2,11],grouploss:[12,14,17,20,21],groupsimilaritydataload:[5,6],ha:[13,16,17,19,21],hadsel:[13,17],half:[13,16],handl:[22,28],hard:[17,21],hash:26,hash_id:5,hashabl:[24,25],have:[7,13,15],head:26,hint:4,how:15,howev:29,http:[12,13,16,17,21],i:[16,29],id:[5,7,8,12,18,20,26],ignor:16,iinfo:29,image_encod:26,implement:[8,16,17,20,21],increas:13,independ:26,index:[0,26],indexing_dataset:[2,3],indexingdataset:4,indexingiterabledataset:4,indic:[13,16,29],individu:[5,26],info:29,info_value_of_dtyp:29,inform:[8,13],initi:[5,26],input:[5,12,13],input_embedding_s:26,instanc:[4,19,22,26],integ:26,intern:22,item:[5,26],iterabledataset:4,itself:15,j:29,jpg:7,json:[13,16,17,19,21],k:29,kei:[13,16,24,25,26],key_extractor:[24,25,26],keyextractortyp:[24,25],keyword:26,kind:26,known:18,kwarg:[5,13,16,26],l2:12,l2_norm:12,label:[5,6,13,16,17,18,21,29],labels_batch:5,layer:26,learn:7,least:13,lecun:[13,17],lemon:[5,7],leonard_nimoy_1:7,leonard_nimoy_2:7,lightn:26,lightningmodul:26,likelihood:16,lime:7,list:[5,8,18,26],load:[13,16,17,19,21,26],loader:[22,26],log:16,logger:26,logit:20,longtensor:[13,14,16,17,20,21],loss:[1,2,5,26],macaroon:7,mai:15,main:[1,2],make:[12,15,16],manhattan:15,map:[24,25,26],margin:[12,13,17,21],mask:29,match:7,matrix:[15,16],max:29,max_value_of_dtyp:29,maximum:29,method:26,metric:[2,5,11,13,14,16,18,19,26],metric_class:[13,16,19],metricmodel:26,might:[8,26],min:29,min_value_of_dtyp:29,mine:[17,21],mini:13,minim:16,minimum:29,modal:16,model:[22,26],modul:[0,1],more:[5,26],muffin:7,multipl:16,multiple_negatives_ranking_loss:[2,11],multiplenegativesrankingloss:16,multipli:16,must:16,name:[13,14,16,17,18,19,21,26],need:16,neg:[13,16,17,21,29],nn:7,non:[24,25,26],none:[4,8,15,22,24,25,26],normal:[12,16],note:16,num_group:[12,20],num_work:[5,24,25],number:[12,20,26,29],obj:[5,7],obj_a:[5,7,16],obj_b:[5,7,16],object:[5,7,8,10,13,15,16,17,18,19,21,22,24,25,26,29],offset:5,onc:5,one:[5,7,17],ones:29,onli:[5,16],onlin:[17,21],online_contrastive_loss:[2,11],onlinecontrastiveloss:17,oper:12,optim:16,option:[5,8,15,17,21,22,24,25,26],orang:[5,7],org:[12,16,21],origin:[4,5,26],original_param:5,other:[7,16,24,25],otherwis:[15,26],output:[5,12,20,26],overridden:8,overwrit:5,overwritten:5,packag:1,page:0,pair:[5,13,16,17,18,29],pairs_count:18,pairssimilaritydataload:5,pairwis:18,pairwise_loss:[2,11],pairwiseloss:[13,16,18],param:[5,13,16,17,19,21],paramet:[5,6,8,12,13,14,15,16,17,18,19,20,21,22,26,29],pass:[5,19,26,29],path:26,pdf:[13,16,17],per:8,perform:[8,22],person:7,pictur:7,pin_memori:5,posit:[13,16,29],possibl:29,pre:[13,14,18,19],pre_collate_fn:[5,8],pre_encoder_col:8,predict:5,prefetch_factor:5,prepar:8,process:[8,22,24,25,26],process_result:26,produc:26,product:[15,20],progress:26,properti:[5,26],provid:26,pseudo:5,publi:[13,17],purpos:[5,13,16,17,19,21],push:[12,17,21],pytorch:[26,29],pytorch_lightn:22,quaterion_model:26,queri:7,question:16,rais:[19,29],ram:26,random:5,rank:16,raw:5,record:6,reduc:13,regular:20,reinforc:4,reinforce_typ:4,repres:[7,29],requir:[4,5,8,24,25],respect:5,restor:26,restrict:4,retriev:[5,16,19,22],routin:22,runtimeerror:19,s:[16,22,26],sampl:[5,13,22],sampler:5,save:[13,16,17,19,21,26],save_serv:26,scalar:[16,17,21],scale:[12,16],score:[5,7,13,16],search:0,second:[7,13,16],see:22,seed:4,send:8,sentenc:16,serializ:[8,13,16,17,19,21],serv:26,set:[24,25],setup_dataload:26,shape:[12,14,15,17,18,20,21,26,29],should:[5,7,8,13,26],shoulder:22,siamesedistancemetr:[13,14,15,18,19],similar:[5,7,16,18,19],similarity_data_load:[2,3],similarity_dataset:[2,3],similarity_loss:[2,11],similarity_metric_nam:16,similarity_sampl:[2,3],similaritydataload:[5,22,26],similaritygroupdataset:6,similaritygroupsampl:[5,6,7],similarityloss:[14,18,19,26],similaritypairsampl:[5,7,16],simpl:6,singl:7,size:[13,14,18,20,24,25,26],size_averag:13,so:16,softmax:[16,20],softmax_loss:[2,11],softmaxloss:20,some:26,sophist:26,sourc:[4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,21,22,24,25,26,28,29],special:5,specif:[5,26],specifi:[16,17,21,26],split:5,squar:[15,17,21],stage:[22,24,25,26,28],standard:6,state:26,store:26,str:[5,8,13,14,16,17,18,19,21,24,25,26,28],strategi:[17,21,29],subgroup:[5,7,13,16,18],submodul:1,subpackag:1,subtyp:4,suitabl:5,sum:13,support:[17,21],sure:15,symmetr:16,t:16,t_co:[4,5],target:[5,26],task:[6,16],temperatur:20,tensor:[5,10,12,13,14,15,16,17,18,19,20,21,24,25,26,29],test:[26,28],test_step:26,text:13,text_encod:26,than:5,thei:26,them:16,thi:[5,7,16,17,26,29],timeout:5,torch:[5,12,17,21,29],train:[1,2,5,8,12,13,14,16,17,18,19,20,21,22,28],train_collat:[2,3],train_dataload:[22,26],trainabl:26,trainable_model:[2,22,23],trainablemodel:[22,24,25,26],traincollat:8,trainer:[22,26],training_step:26,trainstag:[26,28],transform:5,triplet:[17,21,29],triplet_loss:[2,11],tripletloss:21,tupl:[4,5,26],two:[13,14,18,19,22],type:[4,13,16,19,24,25,26,29],typeerror:29,understand:15,unexpect:15,union:[24,25,26,29],uniqu:[5,7],unknown:19,unlik:17,unwrap_cach:26,updat:26,us:[5,13,14,16,17,18,19,21,24,25,26],util:[1,2,5],val_dataload:[22,26],valid:[22,26,28,29],validation_step:26,valu:[12,13,14,16,17,18,20,21,24,25,26,28,29],vector:26,vector_length:[12,14,18,20],version:5,wai:26,want:26,well:26,what:26,when:16,which:[5,6,7,13,26,29],whole:22,within:7,without:5,word:[13,16],work:[5,12,15,16,20],worker:8,worri:16,wrapper:6,x:15,y:15,yann:[13,17],you:[15,16,26],zero:[14,18,20]},titles:["Welcome to Quaterion\u2019s documentation!","quaterion","quaterion package","quaterion.dataset package","quaterion.dataset.indexing_dataset module","quaterion.dataset.similarity_data_loader module","quaterion.dataset.similarity_dataset module","quaterion.dataset.similarity_samples module","quaterion.dataset.train_collater module","quaterion.eval package","quaterion.eval.base_metric module","quaterion.loss package","quaterion.loss.arcface_loss module","quaterion.loss.contrastive_loss module","quaterion.loss.group_loss module","quaterion.loss.metrics module","quaterion.loss.multiple_negatives_ranking_loss module","quaterion.loss.online_contrastive_loss module","quaterion.loss.pairwise_loss module","quaterion.loss.similarity_loss module","quaterion.loss.softmax_loss module","quaterion.loss.triplet_loss module","quaterion.main module","quaterion.train package","quaterion.train.cache package","quaterion.train.cache.cache_config module","quaterion.train.trainable_model module","quaterion.utils package","quaterion.utils.enums module","quaterion.utils.utils module"],titleterms:{"enum":28,arcface_loss:12,base_metr:10,cach:[24,25],cache_config:25,content:[2,3,9,11,23,24,27],contrastive_loss:13,dataset:[3,4,5,6,7,8],document:0,eval:[9,10],group_loss:14,indexing_dataset:4,indic:0,loss:[11,12,13,14,15,16,17,18,19,20,21],main:22,metric:15,modul:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],multiple_negatives_ranking_loss:16,online_contrastive_loss:17,packag:[2,3,9,11,23,24,27],pairwise_loss:18,quaterion:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],s:0,similarity_data_load:5,similarity_dataset:6,similarity_loss:19,similarity_sampl:7,softmax_loss:20,submodul:[2,3,9,11,23,24,27],subpackag:[2,23],tabl:0,train:[23,24,25,26],train_collat:8,trainable_model:26,triplet_loss:21,util:[27,28,29],welcom:0}})