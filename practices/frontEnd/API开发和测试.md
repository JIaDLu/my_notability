## API 开发和测试

在 Postman 中，Body 的不同填写选项主要有以下区别：

**一、form-data**

* 适合上传文件以及键值对数据，通常用于模拟表单提交。可以上传文件，并为每个键值对单独设置参数类型，如文本、文件等。
* 常用于处理包含文件上传的请求以及需要以表单形式提交数据的场景。

**二、x-www-form-urlencoded**

* 会将数据编码为 URL 编码格式，类似于 HTML 表单的默认提交方式。
* 适用于简单的键值对数据提交，数据以 “键 = 值” 的形式进行编码，并使用 “&” 连接多个键值对。

**三、raw**

* 可以输入任意格式的文本数据，如 JSON、XML、纯文本等。
* 需要手动设置正确的 Content-Type 头部信息，以告知服务器所发送数据的格式。常用于发送自定义格式的数据或者与特定 API 交互时需要特定格式的数据。

### 实际案例

#### 烟盒：

## ![13a90e4ec438ab62f25fbe1f28e8c281](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202410092346402.PNG)

> [!TIP]
>
> 这里用form-data是因为我要上传文件
>
> ![image-20241009235205030](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202410092352090.png)
>
> image是flash中写好的要识别的key的name
>
> ```python
> @app.route('/v1/cigarette/display-recognize', methods=['GET', 'POST'])
> def upload_file():
>     file = request.files['image']
>     print(datetime.datetime.now(), file.filename)
>     if file and allowed_file(file.filename):
>         src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
>         file.save(src_path)
>         shutil.copy(src_path, './tmp/ct')
>         image_path = os.path.join('./tmp/ct', file.filename)
>         pid, image_info = core.main.c_main(
>             image_path, current_app.model, file.filename.rsplit('.', 1)[1])
>         if len(image_info) != 0:
>             return jsonify({'err_code': 0,
>                             'err_msg':'',
>                             'image_url': 'http://117.50.189.72:5003/tmp/ct/' + pid,
>                             'draw_url': 'http://117.50.189.72:5003/tmp/draw/' + pid,
>                             'result': image_info})
>         else:
>             return jsonify({'err_code': 1001,
>                             'result': '', 
>                             'err_msg':'未检测到任何目标'})
> ```

#### 糖豆

* 在header中加权限
* 登录接口在请求头中加 checkCaptcha=FaLse  可以绕过验证码，直接用账户密码登录（后端写好的）

![image-20241009235449168](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202410092354224.png)

#### 爬虫

新增接口：客户联系人信息-分页  /platform/qxb/getContactsDataPage   POST传参比之前多了page和limit。群信息-分页getGroupChatPage  POST 参数比之前多了page和limit  （这个之前是get，现在分页用的post，传参放body）

请求传参说的body

![image-20241009235758350](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202410092357418.png)

```python
import json
# instance 1
response = requests.post(
  loginUrl, 
  headers=login_headers, 
  data=json.dumps(login_data)
)
# instance 2
payload = json.dumps({
        "view_id": "vewzXTHe53" 
    })
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer t-g104a9ieUELV6ZNFT2IOUBGU3DUGAHDV7EBLQEGN'
    }
response = requests.request(
  "POST", 
  url, 
  headers=headers, 
  data=payload)
```

