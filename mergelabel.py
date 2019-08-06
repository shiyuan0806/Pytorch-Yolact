import os
import json

with open("instances_train2017",'r') as f:
    content=json.load(f)
    myJson={}
    infoList=[]
    for i, imgInfo in enumerate(content['annotations']):
        if imgInfo['category_id'] not in [1,2,3,4,6,7,8,10,10]:
            continue
        if imgInfo['category_id'] in [2,4]: 
                 imgInfo['category_id']=2 
             if imgInfo['category_id'] in [3,6,7,8]: 
                 imgInfo['category_id']=3 
             if imgInfo['category_id'] in [10,13]: 
                 imgInfo['category_id']=4 
             infoList.append(imgInfo) 
          
         
         myJson['annotations']=infoList 
         myJson['categories']=content['categories'] 
         myJson['licenses']=content['licenses'] 
         myJson['info']=content['info'] 
         myJson['images']=content['images'] 
         needFile=json.dumps(myJson) 
         with open("needJson.json",'w') as f_json: 
             f_json.write(needFile) 
             print("over")


#with open('instances_val2017.json','r') as f: 
#    ...:     content=json.load(f) 
#    ...:     labelCounts={} 
#    ...:     for imgInfo in content['annotations']: 
#    ...:         labelCounts[str(imgInfo['category_id'])]=labelCounts.get(str(imgInfo['category_id']),0)+1 
#    ...:     print(labelCounts) 
#
