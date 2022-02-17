Search.setIndex({docnames:["index","modules","quaterion","quaterion.dataset","quaterion.eval","quaterion.loss","quaterion.train","quaterion.train.encoders","quaterion.utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["index.rst","modules.rst","quaterion.rst","quaterion.dataset.rst","quaterion.eval.rst","quaterion.loss.rst","quaterion.train.rst","quaterion.train.encoders.rst","quaterion.utils.rst"],objects:{"":[[2,0,0,"-","quaterion"]],"quaterion.dataset":[[3,0,0,"-","cache_data_loader"],[3,0,0,"-","similarity_data_loader"],[3,0,0,"-","similarity_samples"]],"quaterion.dataset.cache_data_loader":[[3,1,1,"","CacheDataLoader"]],"quaterion.dataset.cache_data_loader.CacheDataLoader":[[3,2,1,"","batch_size"],[3,3,1,"","cache_collate_fn"],[3,2,1,"","dataset"],[3,2,1,"","drop_last"],[3,3,1,"","fetch_unique_objects"],[3,2,1,"","num_workers"],[3,2,1,"","pin_memory"],[3,2,1,"","prefetch_factor"],[3,2,1,"","sampler"],[3,2,1,"","timeout"]],"quaterion.dataset.similarity_data_loader":[[3,1,1,"","GroupSimilarityDataLoader"],[3,1,1,"","PairsSimilarityDataLoader"],[3,1,1,"","SimilarityDataLoader"]],"quaterion.dataset.similarity_data_loader.GroupSimilarityDataLoader":[[3,2,1,"","batch_size"],[3,3,1,"","collate_fn"],[3,2,1,"","dataset"],[3,2,1,"","drop_last"],[3,3,1,"","fetch_unique_objects"],[3,2,1,"","num_workers"],[3,2,1,"","pin_memory"],[3,2,1,"","prefetch_factor"],[3,2,1,"","sampler"],[3,2,1,"","timeout"]],"quaterion.dataset.similarity_data_loader.PairsSimilarityDataLoader":[[3,2,1,"","batch_size"],[3,3,1,"","collate_fn"],[3,2,1,"","dataset"],[3,2,1,"","drop_last"],[3,3,1,"","fetch_unique_objects"],[3,2,1,"","num_workers"],[3,2,1,"","pin_memory"],[3,2,1,"","prefetch_factor"],[3,2,1,"","sampler"],[3,2,1,"","timeout"]],"quaterion.dataset.similarity_data_loader.SimilarityDataLoader":[[3,2,1,"","batch_size"],[3,2,1,"","dataset"],[3,2,1,"","drop_last"],[3,3,1,"","fetch_unique_objects"],[3,2,1,"","num_workers"],[3,2,1,"","pin_memory"],[3,2,1,"","prefetch_factor"],[3,2,1,"","sampler"],[3,2,1,"","timeout"]],"quaterion.dataset.similarity_samples":[[3,1,1,"","SimilarityGroupSample"],[3,1,1,"","SimilarityPairSample"]],"quaterion.dataset.similarity_samples.SimilarityGroupSample":[[3,2,1,"","group"],[3,2,1,"","obj"]],"quaterion.dataset.similarity_samples.SimilarityPairSample":[[3,2,1,"","obj_a"],[3,2,1,"","obj_b"],[3,2,1,"","score"],[3,2,1,"","subgroup"]],"quaterion.eval":[[4,0,0,"-","base_metric"]],"quaterion.eval.base_metric":[[4,1,1,"","BaseMetric"]],"quaterion.eval.base_metric.BaseMetric":[[4,3,1,"","eval"]],"quaterion.loss":[[5,0,0,"-","arcface_loss"],[5,0,0,"-","contrastive_loss"],[5,0,0,"-","group_loss"],[5,0,0,"-","metrics"],[5,0,0,"-","pairwise_loss"],[5,0,0,"-","similarity_loss"],[5,0,0,"-","softmax_loss"]],"quaterion.loss.arcface_loss":[[5,1,1,"","ArcFaceLoss"]],"quaterion.loss.arcface_loss.ArcFaceLoss":[[5,3,1,"","forward"],[5,2,1,"","training"]],"quaterion.loss.contrastive_loss":[[5,1,1,"","ContrastiveLoss"],[5,4,1,"","relu"]],"quaterion.loss.contrastive_loss.ContrastiveLoss":[[5,3,1,"","forward"],[5,3,1,"","get_config_dict"],[5,3,1,"","metric_class"],[5,2,1,"","training"]],"quaterion.loss.group_loss":[[5,1,1,"","GroupLoss"]],"quaterion.loss.group_loss.GroupLoss":[[5,3,1,"","forward"],[5,2,1,"","training"]],"quaterion.loss.metrics":[[5,1,1,"","SiameseDistanceMetric"],[5,4,1,"","cosine_similarity"],[5,4,1,"","pairwise_distance"]],"quaterion.loss.metrics.SiameseDistanceMetric":[[5,3,1,"","cosine_distance"],[5,3,1,"","dot_product_distance"],[5,3,1,"","euclidean"],[5,3,1,"","manhattan"]],"quaterion.loss.pairwise_loss":[[5,1,1,"","PairwiseLoss"]],"quaterion.loss.pairwise_loss.PairwiseLoss":[[5,3,1,"","forward"],[5,2,1,"","training"]],"quaterion.loss.similarity_loss":[[5,1,1,"","SimilarityLoss"]],"quaterion.loss.similarity_loss.SimilarityLoss":[[5,3,1,"","get_config_dict"],[5,3,1,"","get_distance_function"],[5,3,1,"","metric_class"],[5,2,1,"","training"]],"quaterion.loss.softmax_loss":[[5,1,1,"","SoftmaxLoss"]],"quaterion.loss.softmax_loss.SoftmaxLoss":[[5,3,1,"","forward"],[5,2,1,"","training"]],"quaterion.main":[[2,1,1,"","Quaterion"]],"quaterion.main.Quaterion":[[2,3,1,"","combiner_collate_fn"],[2,3,1,"","fit"]],"quaterion.train":[[6,0,0,"-","cache_mixin"],[7,0,0,"-","encoders"],[6,0,0,"-","trainable_model"]],"quaterion.train.cache_mixin":[[6,1,1,"","CacheMixin"],[6,1,1,"","CacheModel"]],"quaterion.train.cache_mixin.CacheMixin":[[6,2,1,"","CACHE_MULTIPROCESSING_CONTEXT"],[6,3,1,"","cache"]],"quaterion.train.cache_mixin.CacheModel":[[6,3,1,"","predict_dataloader"],[6,3,1,"","predict_step"],[6,3,1,"","test_dataloader"],[6,3,1,"","train_dataloader"],[6,2,1,"","training"],[6,3,1,"","val_dataloader"]],"quaterion.train.encoders":[[7,1,1,"","CacheConfig"],[7,1,1,"","CacheEncoder"],[7,1,1,"","CacheType"],[7,1,1,"","InMemoryCacheEncoder"],[7,0,0,"-","cache_config"],[7,0,0,"-","cache_encoder"],[7,0,0,"-","in_memory_cache_encoder"]],"quaterion.train.encoders.CacheConfig":[[7,2,1,"","batch_size"],[7,2,1,"","cache_type"],[7,2,1,"","key_extractors"],[7,2,1,"","mapping"],[7,2,1,"","num_workers"]],"quaterion.train.encoders.CacheEncoder":[[7,3,1,"","cache_collate"],[7,3,1,"","default_key_extractor"],[7,3,1,"","embedding_size"],[7,3,1,"","fill_cache"],[7,3,1,"","forward"],[7,3,1,"","get_collate_fn"],[7,3,1,"","key_collate_fn"],[7,3,1,"","load"],[7,3,1,"","reset_cache"],[7,3,1,"","save"],[7,3,1,"","trainable"],[7,2,1,"","training"]],"quaterion.train.encoders.CacheType":[[7,2,1,"","AUTO"],[7,2,1,"","CPU"],[7,2,1,"","GPU"]],"quaterion.train.encoders.InMemoryCacheEncoder":[[7,5,1,"","cache_type"],[7,3,1,"","fill_cache"],[7,3,1,"","forward"],[7,3,1,"","get_collate_fn"],[7,3,1,"","reset_cache"],[7,3,1,"","resolve_cache_type"],[7,2,1,"","training"]],"quaterion.train.encoders.cache_config":[[7,1,1,"","CacheConfig"],[7,1,1,"","CacheType"]],"quaterion.train.encoders.cache_config.CacheConfig":[[7,2,1,"","batch_size"],[7,2,1,"","cache_type"],[7,2,1,"","key_extractors"],[7,2,1,"","mapping"],[7,2,1,"","num_workers"]],"quaterion.train.encoders.cache_config.CacheType":[[7,2,1,"","AUTO"],[7,2,1,"","CPU"],[7,2,1,"","GPU"]],"quaterion.train.encoders.cache_encoder":[[7,1,1,"","CacheEncoder"]],"quaterion.train.encoders.cache_encoder.CacheEncoder":[[7,3,1,"","cache_collate"],[7,3,1,"","default_key_extractor"],[7,3,1,"","embedding_size"],[7,3,1,"","fill_cache"],[7,3,1,"","forward"],[7,3,1,"","get_collate_fn"],[7,3,1,"","key_collate_fn"],[7,3,1,"","load"],[7,3,1,"","reset_cache"],[7,3,1,"","save"],[7,3,1,"","trainable"],[7,2,1,"","training"]],"quaterion.train.encoders.in_memory_cache_encoder":[[7,1,1,"","InMemoryCacheEncoder"]],"quaterion.train.encoders.in_memory_cache_encoder.InMemoryCacheEncoder":[[7,5,1,"","cache_type"],[7,3,1,"","fill_cache"],[7,3,1,"","forward"],[7,3,1,"","get_collate_fn"],[7,3,1,"","reset_cache"],[7,3,1,"","resolve_cache_type"],[7,2,1,"","training"]],"quaterion.train.trainable_model":[[6,1,1,"","TrainableModel"]],"quaterion.train.trainable_model.TrainableModel":[[6,3,1,"","configure_caches"],[6,3,1,"","configure_encoders"],[6,3,1,"","configure_head"],[6,3,1,"","configure_loss"],[6,5,1,"","loss"],[6,5,1,"","model"],[6,3,1,"","predict_dataloader"],[6,3,1,"","process_results"],[6,3,1,"","save_servable"],[6,3,1,"","test_dataloader"],[6,3,1,"","test_step"],[6,3,1,"","train_dataloader"],[6,2,1,"","training"],[6,3,1,"","training_step"],[6,3,1,"","val_dataloader"],[6,3,1,"","validation_step"]],"quaterion.utils":[[8,0,0,"-","enums"],[8,0,0,"-","utils"]],"quaterion.utils.enums":[[8,1,1,"","TrainStage"]],"quaterion.utils.enums.TrainStage":[[8,2,1,"","TEST"],[8,2,1,"","TRAIN"],[8,2,1,"","VALIDATION"]],"quaterion.utils.utils":[[8,4,1,"","info_value_of_dtype"],[8,4,1,"","max_value_of_dtype"],[8,4,1,"","min_value_of_dtype"]],quaterion:[[3,0,0,"-","dataset"],[4,0,0,"-","eval"],[5,0,0,"-","loss"],[2,0,0,"-","main"],[6,0,0,"-","train"],[8,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"],"5":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function","5":"py:property"},terms:{"0":[3,5,6],"05":5,"06":5,"07698":5,"0x106bc90e0":6,"1":[3,5,6],"10":3,"100":5,"11":3,"128":5,"1801":5,"1e":5,"1st_pair_1st_obj":3,"1st_pair_2nd_obj":3,"2":[3,5],"209":3,"2nd_pair_1st_obj":3,"2nd_pair_2nd_obj":3,"3":3,"32":7,"371":3,"5":[5,6],"555":3,"6":3,"64":5,"7":3,"8":[3,5],"9":3,"case":[3,6,7],"class":[2,3,4,5,6,7,8],"default":[5,6,7],"do":6,"enum":[1,2,6,7],"float":[3,5,8],"function":[2,3,5,6,7],"int":[3,5,6,7,8],"return":[2,3,5,6,7,8],"static":[5,7],"true":[5,6],"while":3,A:[2,6],For:6,If:[5,6,7],In:[3,6,7],It:[2,5,6,7],The:[5,6],There:6,_2:5,__batch_of_encoder__:3,__init__:3,ab:5,abl:7,about:8,abov:6,accord:[6,7],across:3,ad:6,add:6,addit:[5,6],affect:7,after:7,all:[3,5,6,7],allow:[6,8],along:5,alreadi:7,also:[6,7],an:[5,6],angular:5,ani:[2,3,5,6,7],anoth:3,apart:5,appl:3,appli:[6,7],ar:[5,6,7],arbitrari:6,arcface_loss:[1,2],arcfaceloss:5,arg:6,argument:[3,5,6],arxiv:5,assembl:2,assign:6,associ:5,attribut:5,auto:[6,7],avail:[5,7],averag:5,avoid:[3,5,6,7],bar:6,barak_obama_1:3,barak_obama_2:3,barak_obama_3:3,base:[2,3,4,5,6,7,8],base_metr:[1,2],basemetr:4,batch:[2,3,5,6,7],batch_cifar:6,batch_idx:6,batch_mnist:6,batch_sampl:3,batch_siz:[3,5,6,7],been:7,befor:6,between:5,bool:[3,5,6,7,8],broadcast:5,built:7,cach:[2,3,6,7],cache_col:7,cache_collate_fn:3,cache_config:[2,6],cache_data_load:[1,2],cache_encod:[2,6],cache_mixin:[1,2],cache_multiprocessing_context:6,cache_typ:[6,7],cachecollatefntyp:7,cacheconfig:[6,7],cached_encoders_collate_fn:3,cachedataload:[3,7],cachedencod:7,cacheencod:[6,7],cachemixin:6,cachemodel:6,cachetyp:[6,7],calcul:[3,5,6,7],call:5,callabl:[2,3,5,7],can:[5,6,7],candi:3,cannot:5,cdot:5,checkpoint:[6,7],cheesecak:3,chopra:5,chosen:5,cifar:6,cifar_load:6,cl:5,classmethod:[2,3,5,6,7],closer:3,collat:[2,3,6,7],collate_fn:3,collect:[3,6,7],com:5,combin:2,combiner_collate_fn:2,common:5,complic:7,compos:6,comput:[5,6],config:[5,6],configur:6,configure_cach:[6,7],configure_encod:6,configure_head:6,configure_loss:6,construct:3,contain:[5,6],content:1,context:3,contrast:5,contrastive_loss:[1,2],contrastiveloss:5,conveni:[3,6],convert:[2,7],core:6,correct:6,correspond:[5,6,7],cosin:5,cosine_dist:5,cosine_similar:5,cpu:[6,7],cross:5,csv:3,cuda:7,current:[3,5,6,7],custom:7,data:[2,3,6,7,8],dataload:[2,3,6],dataloader_idx:6,dataset:[1,2,5,6],def:6,default_encoder_kei:6,default_key_extractor:7,defin:[5,6,7],design:5,determinist:7,devic:[6,7],dfrac:5,dict:[2,3,5,6,7],differ:3,dim:5,dimens:5,directori:7,displai:6,distanc:5,distance_metric_nam:5,distinguish:5,distribut:6,divid:5,divis:5,doe:[7,8],don:[6,7],dot:5,dot_product_dist:5,download:6,drop_last:3,dtype:8,dure:[2,6,7],dwarf:2,e:6,each:[3,6],effect:5,either:5,elon_musk_1:3,elon_musk_2:3,elon_musk_3:3,els:7,embed:[3,5,6,7],embedding_s:[5,6,7],encapsul:6,encod:[2,3,5,6],encoder_head:6,encoder_nam:[3,7],encoderhead:6,entir:5,entropi:5,ep:5,epoch:6,epsilon:5,especi:5,etc:6,euclidean:5,eval:[1,2],evalu:6,exampl:[3,5,6],exdb:5,expect:5,extract:[2,3,7],extractor:7,f:5,face:3,factori:7,fals:[3,5,6],farther:2,featur:[2,3],features_col:2,fetch:3,fetch_unique_object:3,fewer:5,file_nam:3,fill:[6,7],fill_cach:7,finfo:8,first:[3,5],fit:[2,6],flag:7,flat:5,follow:6,fork:6,forward:[5,7],from:[2,3,5,7],frozen:7,function_nam:5,further:[3,5],g:6,gener:[3,5,7],genuin:6,get:6,get_collate_fn:7,get_config_dict:5,get_distance_funct:5,giant:2,given:8,gpu:[6,7],grapefruit:3,group:[3,5],group_id:3,group_loss:[1,2],grouploss:5,groupsimilaritydataload:3,ha:[5,7],hadsel:5,half:5,handl:[2,8],happen:6,hardwar:6,hash:[6,7],hashabl:[3,6,7],have:[3,5,6],head:6,here:6,how:5,howev:6,http:5,id:[3,5,6],iinfo:8,image_encod:6,implement:[5,6,7],in_memory_cache_encod:[2,6],increas:5,independ:6,index:[0,6],indic:5,infer:7,info:8,info_value_of_dtyp:8,inform:5,inher:7,inherit:6,initi:6,inmemorycacheencod:7,input1:5,input2:5,input:[2,5,6,7],input_embedding_s:6,input_path:7,instanc:[2,5,6],instanti:7,integ:6,intern:2,item:6,iter:[6,7],itself:5,jpg:3,json:5,kei:[3,5,6,7],key_collate_fn:7,key_extractor:[3,6,7],keyword:6,kind:6,known:5,kwarg:[3,5,6],label:[2,3,5],labels_col:2,lambda:6,layer:[6,7],learn:3,least:5,lecun:5,lemon:3,leonard_nimoy_1:3,leonard_nimoy_2:3,lightn:6,lightningmodul:6,lime:3,list:[2,3,5,6,7],load:[3,5,6,7],loader:6,loader_a:6,loader_b:6,loader_n:6,logger:6,logic:6,logit:5,longtensor:5,loss:[1,2,6],macaroon:3,mai:5,main:1,make:[5,6],manag:6,mandarin:3,manhattan:5,map:[3,6,7],margin:5,match:[3,6],matrix:5,max:[5,8],max_value_of_dtyp:8,maximum:8,mess:6,method:[6,7],metric:[1,2,6],metric_class:5,metricmodel:6,min:8,min_value_of_dtyp:8,mini:5,minimum:8,mnist:6,mnist_load:6,mock:6,model:[2,6,7],modul:[0,1],more:6,muffin:3,multipl:6,multiprocessing_context:3,must:5,name:[5,6,7],necessari:6,need:6,neg:5,nn:[3,5],non:7,none:[2,3,6,7],normal:6,num:6,num_group:5,num_work:[3,7],number:5,nutella:3,obj:[3,6,7],obj_a:3,obj_b:3,object:[2,3,4,5,6,7,8],one:[3,6,7],onli:[3,6,7],option:[2,3,5,6,7],orang:3,order:6,org:5,origin:[2,6],other:[3,7],otherwis:[3,6,7],output:[5,6,7],output_path:7,packag:1,page:[0,6],pair:[3,5],pairs_count:5,pairssimilaritydataload:3,pairwis:5,pairwise_dist:5,pairwise_loss:[1,2],pairwiseloss:5,param:5,paramet:[2,3,5,6,7,8],paramref:6,pass:[3,5,6,8],path:[6,7],pattern:6,pdf:5,perform:[2,6],persist:7,persistent_work:3,person:3,pictur:3,pin_memori:3,pleas:6,posit:[5,6],postiv:6,pre:5,predict:6,predict_dataload:6,predict_step:6,prefetch_factor:3,prepar:6,prepare_data:6,preserv:6,print:5,process:[2,3,6,7],process_result:6,produc:6,product:5,progress:6,promot:5,properti:[6,7],provid:[3,6,7],publi:5,purpos:5,push:5,pytorch:[6,8],pytorch_lightn:[2,6],quaterion_model:[6,7],queri:3,rais:[5,8],randn:5,raw:[2,3,7],readi:6,receiv:7,recommend:6,reduc:5,refer:5,regular:5,reload:6,reload_dataloaders_every_n_epoch:6,relu:5,repeat:[3,7],repres:3,requir:[6,7],reset:7,reset_cach:7,resolv:7,resolve_cache_typ:7,respons:2,result:[3,5,7],retriev:[2,5,7],reus:7,root:6,routin:2,runtimeerror:5,s:[2,3,6,7],same:6,sampl:[2,3,5,6],sampler:[3,6],save:[5,6,7],save_serv:6,scale:5,score:[3,5],search:0,second:[3,5],see:[2,5,6],self:6,sequenc:[3,6],serializ:5,serv:6,set:[6,7],setup:6,shape:[5,6,7],should:[3,5,6],shoulder:2,shuffl:[3,6],siamesedistancemetr:5,similar:[3,5,6],similarity_data_load:[1,2,6],similarity_loss:[1,2,6],similarity_sampl:[1,2],similaritydataload:[3,6],similaritygroupsampl:3,similarityloss:[5,6],similaritypairsampl:3,singl:[3,6,7],situat:7,size:[5,6,7],size_averag:5,small:5,softmax:5,softmax_loss:[1,2],softmaxloss:5,some:6,specif:6,specifi:[6,7],split:6,squeez:5,stage:[2,6,7,8],stai:7,state:[6,7],store:7,str:[2,3,5,6,7,8],stuff:6,subgroup:[3,5],submodul:1,subpackag:1,suitabl:[2,7],sum:5,support:[5,7],sure:5,t:[3,6,7],t_co:3,target:6,temperatur:5,tensor:[2,3,4,5,6,7],tensorinterchang:[2,7],test:[6,8],test_dataload:6,test_step:6,text:5,text_encod:6,thei:6,them:[3,6],thi:[3,5,6,7],time:7,timeout:3,torch:[2,3,4,5,6,7,8],totensor:6,train:[1,2,5,8],train_dataload:[2,6],trainabl:7,trainable_model:[1,2],trainablemodel:[2,6,7],trainer:[2,6],training_step:6,trainstag:[6,8],transform:6,tupl:[2,3,6,7],two:[2,5],type:[2,3,5,6,7,8],typeerror:8,understand:5,unexpect:5,union:[2,3,6,7,8],uniqu:3,unique_objects_extractor:3,unknown:5,unless:6,untouch:7,us:[3,5,6,7],util:[1,2,3,6],val:6,val_dataload:[2,6],valid:[2,6,8],validation_step:6,valu:[3,5,6,7,8],vector_length:5,vert:5,wai:3,well:6,what:6,when:7,where:6,whether:7,which:[3,5,6,7,8],whole:2,within:3,without:3,word:[3,5],work:5,worker:6,worker_init_fn:3,wrap:7,wrapper:7,x1:5,x2:5,x:[5,7],x_1:5,x_2:5,y:5,yann:5,you:[5,6],yourself:6,zero:5},titles:["Welcome to Quaterion\u2019s documentation!","quaterion","quaterion package","quaterion.dataset package","quaterion.eval package","quaterion.loss package","quaterion.train package","quaterion.train.encoders package","quaterion.utils package"],titleterms:{"enum":8,arcface_loss:5,base_metr:4,cache_config:7,cache_data_load:3,cache_encod:7,cache_mixin:6,content:[2,3,4,5,6,7,8],contrastive_loss:5,dataset:3,document:0,encod:7,eval:4,group_loss:5,in_memory_cache_encod:7,indic:0,loss:5,main:2,metric:5,modul:[2,3,4,5,6,7,8],packag:[2,3,4,5,6,7,8],pairwise_loss:5,quaterion:[0,1,2,3,4,5,6,7,8],s:0,similarity_data_load:3,similarity_loss:5,similarity_sampl:3,softmax_loss:5,submodul:[2,3,4,5,6,7,8],subpackag:[2,6],tabl:0,train:[6,7],trainable_model:6,util:8,welcom:0}})