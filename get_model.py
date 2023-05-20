from flask import Flask, request, jsonify, render_template
import pickle

# 创建Flask应用程序
app = Flask(__name__)

# 加载训练好的模型
with open('MLP.model', 'rb') as f:
    model = pickle.load(f)

# 定义路由和视图函数
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 获取用户输入的数据
    features = request.form.values()

    # 进行预测
    prediction = model.predict([features])

    # 返回预测结果
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
