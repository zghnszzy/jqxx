import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(file_name):
    """加载数据文件，如果不存在则报错"""
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"数据文件 {file_name} 未找到，请确保它在当前目录下")
    return pd.read_csv(file_name)

try:
    # 数据加载 - 使用相对路径或完整路径
    print("正在加载数据文件...")
    train_data = load_data('train.csv')
    test_data_a = load_data('testA.csv')
    test_data_b = load_data('sample_submit.csv')
    print("数据加载成功")
except FileNotFoundError as e:
    print(e)
    print("请确保以下文件存在于当前目录中:")
    print("- train.csv")
    print("- test_a.csv")
    print("- test_b.csv")
    exit(1)

def preprocess_data(df):
    # 复制数据避免修改原始数据
    df = df.copy()
    
    # 处理employmentLength - 转换为数值
    if 'employmentLength' in df.columns:
        df['employmentLength'] = df['employmentLength'].replace({
            '< 1 year': 0,
            '1 year': 1,
            '2 years': 2,
            '3 years': 3,
            '4 years': 4,
            '5 years': 5,
            '6 years': 6,
            '7 years': 7,
            '8 years': 8,
            '9 years': 9,
            '10+ years': 10
        })
    
    # 处理日期特征 - 转换为时间戳或提取年份月份
    if 'issueDate' in df.columns:
        df['issueDate'] = pd.to_datetime(df['issueDate'])
        df['issueDate_year'] = df['issueDate'].dt.year
        df['issueDate_month'] = df['issueDate'].dt.month
    
    if 'earliesCreditLine' in df.columns:
        df['earliesCreditLine'] = pd.to_datetime(df['earliesCreditLine'], errors='coerce')
        df['earliesCreditLine_year'] = df['earliesCreditLine'].dt.year
        df['earliesCreditLine_month'] = df['earliesCreditLine'].dt.month
    
    # 计算fico平均分（如果存在相关列）
    if all(col in df.columns for col in ['ficoRangeLow', 'ficoRangeHigh']):
        df['ficoAvg'] = (df['ficoRangeLow'] + df['ficoRangeHigh']) / 2
    
    # 删除原始日期列（如果存在）
    df.drop(['issueDate', 'earliesCreditLine'], axis=1, inplace=True, errors='ignore')
    
    # 处理分类变量 - 强制转换为字符串并填充缺失值
    cat_cols = ['grade', 'subGrade', 'employmentTitle', 'homeOwnership', 
               'verificationStatus', 'purpose', 'postCode', 'regionCode',
               'initialListStatus', 'applicationType', 'title', 'policyCode']
    
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)  # 强制转换为字符串
            df[col].fillna('missing', inplace=True)
    
    return df

# 预处理训练数据和测试数据
print("开始数据预处理...")
train_data = preprocess_data(train_data)
test_data_a = preprocess_data(test_data_a)
test_data_b = preprocess_data(test_data_b)
print("数据预处理完成")

# 分离特征和目标变量
if 'isDefault' not in train_data.columns:
    raise KeyError("训练数据中缺少目标列 'isDefault'")

X = train_data.drop(['id', 'isDefault'], axis=1, errors='ignore')
y = train_data['isDefault']

# 定义数值型和类别型特征
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# 创建预处理管道
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 定义模型
models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    'LogisticRegression': LogisticRegression(C=0.1, solver='saga', max_iter=1000, random_state=42, n_jobs=-1)
}

# 训练和评估模型
best_model = None
best_auc = 0

print("\n开始模型训练...")
for name, model in models.items():
    # 创建完整管道
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', model)])
    
    # 训练模型
    print(f"正在训练 {name}...")
    pipeline.fit(X_train, y_train)
    
    # 预测验证集
    y_pred = pipeline.predict_proba(X_val)[:, 1]
    
    # 计算AUC
    auc = roc_auc_score(y_val, y_pred)
    print(f'{name} AUC: {auc:.4f}')
    
    # 更新最佳模型
    if auc > best_auc:
        best_auc = auc
        best_model = pipeline

# 使用最佳模型预测测试集
# 修改predict_and_save函数
def predict_and_save(test_data, filename):
    try:
        # 确保特征与训练集一致
        missing_cols = set(X_train.columns) - set(test_data.columns)
        if missing_cols:
            print(f"警告: 自动填充缺失列 {missing_cols}")
            for col in missing_cols:
                test_data[col] = 0  # 数值列填0，分类列填'missing'
        
        ids = test_data['id']
        X_test = test_data[X_train.columns]  # 严格按训练集顺序
        
        y_pred = best_model.predict_proba(X_test)[:, 1]
        pd.DataFrame({'id': ids, 'isDefault': y_pred}).to_csv(filename, index=False)
        print(f"成功保存到 {filename}")
        
    except Exception as e:
        print(f"保存失败: {str(e)}")
# 特征重要性分析（如果使用树模型）
if best_model and hasattr(best_model.named_steps['model'], 'feature_importances_'):
    try:
        # 获取特征名称（包括one-hot编码后的）
        ohe_columns = list(best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols))
        all_features = numerical_cols + ohe_columns
        
        # 获取特征重要性
        importances = best_model.named_steps['model'].feature_importances_
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({'feature': all_features, 'importance': importances})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # 打印最重要的20个特征
        print("\nTop 20重要特征:")
        print(feature_importance.head(20))
    except Exception as e:
        print(f"\n无法获取特征重要性: {e}")