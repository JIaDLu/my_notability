## 接口测试

HTTP请求处理器，用于根据输入参数动态地发送HTTP请求。

接口测试请求通常需要：

1. 请求方法

2. 请求的URL

3. 请求体（body）例如：POST请求的JSON格式请求体：

   ![image-20240909145419516](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202409091456012.png)

   ​	{
    	 "username": "jayden",
     	"email": "jayden@example.com"
   ​	}

4. 查询信息：附加在URL后的键值对，用于向服务器传递额外的信息。常见于GET请求中。

   ![image-20240909145758825](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202409091501726.png)

5. 请求头：是HTTP请求的一部分，包含了一系列的键值对，用于传递客户端给服务器的额外信息。请求头可以告诉服务器有关客户端的身份、请求的类型、数据格式等内。

   *  **身份识别**：一些API或服务器要求请求request提供认证信息（如`Authorization`头）以确保只有经过验证的用户才能访问。

   * **指定数据格式**：通过`Content-Type`头，客户端告诉服务器发送的数据类型，比如JSON、XML等。服务器可以根据这个信息解析请求体。`Content-Type: application/json` 表示请求体的数据是JSON格式。

   * **用户代理信息**（你的设备信息）：`User-Agent`头可以标识发起请求的客户端（如浏览器类型、操作系统等），方便服务器根据客户端类型返回不同的内容。（**<u>爬虫会经常需要加这个，用来模拟浏览器请求</u>**）
     * 例子：`User-Agent: Mozilla/5.0` 告诉服务器请求来自某个特定版本的浏览器。


```python
    headers = {"Content-Type":"application/json"}
    if inputParam.header:
        headers = json.loads(inputParam.header)

    if 'User-Agent' not in headers.keys():
        headers['User-Agent'] = 'Coze/1.0.0 (https://coze.com)'

    if "Content-Type" not in headers.keys():
        headers['Content-Type'] = 'application/json'
```

​	默认hearders是`"Content-Type": "application/json"`。

​	如果输入参数中包含`header`，则将其解析为字典形式。

​	如果请求头中没有`User-Agent`，它会自动添加一个默认的`User-Agent`（“Coze/1.0.0”）。

​	如果请求头中没有`Content-Type`，也会设置为`application/json`。

![image-20240909130556474](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202409091305680.png)

```python
from runtime import Args
from typings.api.api import Input, Output
import requests
import json
from datetime import datetime, timedelta
from html import unescape

def handler(args: Args[Input])->Output: 

    inputParam = args.input  
    host = inputParam.host     # inputParam.host 获取输入参数中的主机地址。

    url = 'https://' + host + '/tangpang/api/meinian/getData'  

    headers = {"Content-Type":"application/json"}
    if inputParam.header:
        headers = json.loads(inputParam.header)

    if 'User-Agent' not in headers.keys():
        headers['User-Agent'] = 'Coze/1.0.0 (https://coze.com)'

    if "Content-Type" not in headers.keys():
        headers['Content-Type'] = 'application/json'


    
    patientId = inputParam.patientId

    # 不必填
    if inputParam.hours:
        hourCount = inputParam.hours
    else:
        hourCount = 24

    
    current_time = datetime.now()
    past_time = current_time - timedelta(hours=hourCount)
    
    # 转换为时间戳
    current_timestamp = int(current_time.timestamp() * 1000)
    past_timestamp = int(past_time.timestamp() * 1000)
 

    payload = {
        "ak": "MN_NDTP_SUP",
        "sk": "u9fwz96eoj5143n54ybx3egjpvr40bcpuciqfistesft22azewbk1c6bnittp5j0",
        "url": "https://openapi.health-100.cn/chronicdisease-manage/api/v1/cgmData/cgmStatistics",
        "methodName": "post",
        "params": {
            "patientId": patientId,
            "beginT": past_timestamp,
            "endT": current_timestamp
	    }
    }

    payload = json.dumps(payload)

    url = unescape(url)  # 使用unescape解码URL中的HTML实体。这可以避免URL中的特殊字符被误解为HTML实体。

    res = None
    res = requests.post(url, data=payload, headers=headers)
    resJson = res.json()
    
    if resJson["code"] == 0:
        resData = resJson["data"]
        
        if resData["tir"] is None:
             ret: Output = {
                "message": "未检测到数据"
            }
        else:
            ret: Output = {
                "message": "success",
                "tir": resData["tir"],
                "tir_status": resData["tirStatus"],
                "tar": resData["tar"],
                "tar_status": resData["tarStatus"],
                "tbr": resData["tbr"],
                "tbr_status":resData["tbrStatus"],
                "mg": resData["mg"],
                "mg_status": resData["mgStatus"],
                "ehba": resData["ehba"],
                "ehba_status": resData["ehbaStatus"],
                "cv": resData["cv"],
                "cv_status": resData["cvStatus"],
                "mage":resData['mage'],
                "mage_status":resData["mageStatus"],
                "min":resData["min"],
                "minTime":resData["minStatus"],
                "max":resData["max"],
                "maxTime":resData["maxStatus"]
            }

    return ret
```

