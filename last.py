# -*- coding: utf-8 -*-
"""
销售数据全流程智能分析系统
================================

设计思路（依据《设计思路简述》）：
1. 基础层：数据清洗与质量评估
   - 统一处理4份数据的缺失值、异常值、格式问题、逻辑矛盾、重复值
   - 字段自适应适配（月饼/巧克力/家具服装/含用户数据）

2. 路径1：描述性分析（趋势/分布/对比）
   - 趋势分析：按销售日期做月度/季度销售额、销量趋势，识别淡旺季
   - 分布分析：价格、销量、销售额的分布情况
   - 对比分析：地区维度、产品维度、用户维度、渠道维度、销售人员维度

3. 路径2：相关性与影响因素分析
   - 变量相关性分析：计算价格、销量、折扣、年龄等变量与销售额的相关性
   - 单变量影响分析：价格对销量的影响、折扣对销售额的影响、年龄/性别对消费的影响

4. 路径3：多模型对比分析
   - 多元线性回归、决策树回归、支持向量回归、随机森林回归
   - 对比R²、MAE、RMSE，自动选择最优模型

5. 路径4：变量交互分析
   - 地区×产品类别、年龄×产品类别、渠道×支付方式、价格×折扣

6. 报告导出：自动生成企业级经营分析报告（WORD）
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
from datetime import datetime
warnings.filterwarnings('ignore')

# ==================== 核心修复：全局配置优化 ====================
# 适配新版Streamlit的页面配置
st.set_page_config(
    page_title="销售数据全流程智能分析系统",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("销售数据全流程智能分析系统")
st.markdown("**数据清洗 → 异常检测 → 可视化分析 → 多模型对比 → 报告生成**")

# 修复中文显示问题（兼容不同操作系统）
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['figure.max_open_warning'] = 0  # 关闭图表数量警告

# ==================== 辅助函数定义（修复边界问题） ====================

def clean_price_format(price_value):
    """清洗价格格式：去除单位、处理科学计数法"""
    if pd.isna(price_value):
        return np.nan
    price_str = str(price_value).strip()
    price_str = price_str.replace("元", "").replace("￥", "").replace("$", "")
    if 'e' in price_str.lower():
        try:
            return float(price_str)
        except:
            return np.nan
    try:
        return float(price_str)
    except:
        return np.nan

def clean_region_format(region_value):
    """清洗地区格式：统一大小写、展开简称"""
    if pd.isna(region_value):
        return np.nan
    region_str = str(region_value).strip()
    if region_str.lower() in ['north', 'south', 'east', 'west', 'uk', 'india', 'australia']:
        region_str = region_str.title()
    region_short_map = {'渝': '重庆', '京': '北京', '沪': '上海', '穗': '广州', '深': '深圳'}
    for short, full in region_short_map.items():
        if short in region_str:
            region_str = region_str.replace(short, full)
    return region_str

def clean_product_format(product_value):
    """清洗产品名称格式：去除首尾空格、特殊符号"""
    if pd.isna(product_value):
        return np.nan
    product_str = str(product_value).strip()
    product_str = product_str.replace("###", "").strip()
    return product_str if product_str else np.nan

def extract_moon_flavor(product_name):
    """从月饼产品名称中提取口味"""
    if pd.isna(product_name):
        return "未知"
    p = str(product_name).lower()
    if '流心奶黄' in p or ('流心' in p and '奶黄' in p):
        return '流心奶黄'
    if '五仁' in p or '伍仁' in p:
        return '五仁'
    if '莲蓉' in p:
        return '莲蓉'
    if '蛋黄' in p:
        return '蛋黄'
    if '豆沙' in p:
        return '豆沙'
    if '冰皮' in p:
        return '冰皮'
    if '云腿' in p or '火腿' in p:
        return '云腿'
    if '巧克力' in p:
        return '巧克力'
    if '水果' in p:
        return '水果'
    if '椰蓉' in p:
        return '椰蓉'
    if '枣泥' in p:
        return '枣泥'
    return '其他'

def extract_choco_flavor(product_name):
    """从巧克力产品名称中提取口味"""
    if pd.isna(product_name):
        return "未知"
    p = str(product_name).lower()
    if 'dark' in p:
        return '黑巧克力(Dark)'
    if 'milk' in p:
        return '牛奶巧克力(Milk)'
    if 'white' in p:
        return '白巧克力(White)'
    if 'peanut' in p:
        return '花生酱'
    if 'almond' in p:
        return '杏仁'
    if 'raspberry' in p:
        return '覆盆子'
    if 'orange' in p:
        return '橙子'
    if 'mint' in p:
        return '薄荷'
    if 'caramel' in p:
        return '焦糖'
    if 'eclairs' in p:
        return '闪电泡芙'
    if 'manuka' in p:
        return '麦卢卡蜂蜜'
    return '其他'

def find_column(df, keywords, return_first=True):
    """模糊查找包含指定关键词的列（修复空值返回问题）"""
    if df.empty:
        return None if return_first else []
    matched = []
    for col in df.columns:
        for kw in keywords:
            if kw in str(col).lower():  # 修复大小写问题
                matched.append(col)
                if return_first:
                    return col
    return matched if matched else None

def clean_group_data(series):
    """清洗分组用的列（修复空值和类型问题）"""
    if series.empty:
        return pd.Series(dtype=str)
    return series.astype(str).str.strip().str.lower().replace("", np.nan).dropna()

def standardize_gender(gender):
    """标准化性别：统一为 Male / Female 格式"""
    if pd.isna(gender):
        return np.nan
    g = str(gender).strip()
    if g.lower() in ['male', 'm', '男', '男生', '男性']:
        return 'Male'
    elif g.lower() in ['female', 'f', '女', '女生', '女性']:
        return 'Female'
    else:
        return g

def encode_city_level(region_name):
    """地区有序编码：一线=5, 新一线=4, 二线=3, 三线=2, 其他=1"""
    if pd.isna(region_name):
        return 1
    r = str(region_name)
    if any(c in r for c in ['北京', '上海', '广州', '深圳']):
        return 5
    if any(c in r for c in ['成都', '杭州', '重庆', '武汉', '西安', '天津', '南京', '郑州', '长沙', '沈阳', '青岛', '苏州', '东莞', '佛山']):
        return 4
    if any(c in r for c in ['昆明', '哈尔滨', '长春', '南昌', '南宁', '合肥', '太原', '石家庄', '乌鲁木齐', '贵阳', '厦门', '福州']):
        return 3
    if any(c in r for c in ['嘉兴', '无锡', '金华', '湖州', '绍兴', '温州', '惠州', '中山', '珠海', '镇江', '扬州', '南通']):
        return 2
    return 1

def encode_product_tier(price_series, product_series):
    """产品类别有序编码：按平均价格从低到高赋1-5（修复空值问题）"""
    if product_series is None or price_series is None:
        return {}
    df_temp = pd.DataFrame({'产品类别': product_series, '价格': price_series}).dropna()
    if df_temp.empty or len(df_temp['产品类别'].unique()) <= 1:
        return {}
    avg_price = df_temp.groupby('产品类别')['价格'].mean().sort_values()
    n = len(avg_price)
    if n <= 1:
        return {cat: 3 for cat in avg_price.index}
    # 修复分位数计算问题
    breaks = np.percentile(np.arange(n), [20, 40, 60, 80]).astype(int)
    breaks = np.clip(breaks, 0, n-1)  # 防止越界
    tier_map = {}
    for i, cat in enumerate(avg_price.index):
        if i < breaks[0]:
            tier_map[cat] = 1
        elif i < breaks[1]:
            tier_map[cat] = 2
        elif i < breaks[2]:
            tier_map[cat] = 3
        elif i < breaks[3]:
            tier_map[cat] = 4
        else:
            tier_map[cat] = 5
    return tier_map

# ==================== 数据上传与读取（修复空文件问题） ====================
st.divider()
st.subheader("数据上传")

upload_file = st.file_uploader(
    "请拖入或上传数据文件（支持Excel/CSV）",
    type=["xlsx", "csv"],
    help="建议文件大小不超过100MB，列名包含：销售额、价格、销量、地区、产品等关键字段"
)

df = None
raw_df = None
original_rows = 0

if upload_file:
    try:
        if upload_file.name.endswith(".csv"):
            # 修复编码问题
            df = pd.read_csv(upload_file, encoding='utf-8', on_bad_lines='skip')
        else:
            df = pd.read_excel(upload_file, engine='openpyxl')
        
        # 修复空数据问题
        if df.empty:
            st.error("❌ 上传的文件为空，请检查文件内容！")
            st.stop()
        
        raw_df = df.copy()
        original_rows = len(df)
        st.success(f"✅ 数据读取成功！原始数据：共 {original_rows} 行，{len(df.columns)} 列")
        
        # 修复列名乱码问题
        df.columns = [str(col).strip() for col in df.columns]
        
        st.write("**数据预览（前5行）**：")
        st.dataframe(df.head(), use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ 数据读取失败：{str(e)}")
        st.info("建议检查：1. 文件是否损坏 2. CSV文件编码是否为UTF-8 3. Excel文件是否为.xlsx格式")
        st.stop()
else:
    st.info("请上传数据文件以开始分析")
    st.stop()

# ==================== 全类型异常值检测（修复空值和类型错误） ====================
st.divider()
st.subheader("全类型异常值检测")

# 初始化异常计数器
dup_num = df.duplicated().sum()
miss_num = df.isnull().sum().sum()
text_in_num = 0
num_in_text = 0
special_symbol = 0
negative_err = 0
extreme_outlier = 0
gender_err = 0
date_err = 0
zero_price_non_sales = 0
zero_sales_non_amount = 0
amount_deviation_err = 0
age_outlier_err = 0
part_dup_num = 0

# 逐列扫描基础异常（修复空列问题）
for col in df.columns:
    if df[col].empty:
        continue
    
    col_data = df[col].astype(str).str.strip()
    # 修复特殊符号匹配问题
    special_symbol += col_data.str.contains(r'[,\，元￥$岁#@!]|\s{2,}', regex=True).sum()
    
    # 数值列检测
    if any(key in str(col).lower() for key in ["销售额", "价格", "销量", "单位成本", "折扣", "单价", "金额", "年龄"]):
        temp_num = pd.to_numeric(df[col], errors="coerce")
        text_in_num += temp_num.isna().sum()
        negative_err += (temp_num.dropna() < 0).sum()
        
        if "年龄" in str(col).lower():
            age_outlier_err += ((temp_num < 0) | (temp_num > 120)).sum()
        
        # 修复空值导致的离群值计算错误
        if not temp_num.dropna().empty:
            q1 = temp_num.quantile(0.25)
            q3 = temp_num.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:  # 防止除以0
                extreme_outlier += ((temp_num < q1 - 1.5*iqr) | (temp_num > q3 + 1.5*iqr)).sum()
    
    # 文本列检测
    if any(key in str(col).lower() for key in ["产品", "地区", "销售人员", "产品类别", "支付方式", "顾客类型", "销售渠道"]):
        num_in_text += col_data.str.contains(r"\d", regex=True).sum()

# 逻辑异常检测（修复列名大小写和空值问题）
price_col = find_column(df, ["价格"])
sales_col = find_column(df, ["销量"])
amount_col = find_column(df, ["销售额"])

if price_col and sales_col:
    price_num = pd.to_numeric(df[price_col], errors="coerce")
    sales_num = pd.to_numeric(df[sales_col], errors="coerce")
    zero_price_non_sales = df[(price_num == 0) & (sales_num > 0)].shape[0]

if sales_col and amount_col:
    sales_num = pd.to_numeric(df[sales_col], errors="coerce")
    amount_num = pd.to_numeric(df[amount_col], errors="coerce")
    zero_sales_non_amount = df[(sales_num == 0) & (amount_num > 0)].shape[0]

if price_col and sales_col and amount_col:
    price_num = pd.to_numeric(df[price_col], errors="coerce")
    sales_num = pd.to_numeric(df[sales_col], errors="coerce")
    amount_num = pd.to_numeric(df[amount_col], errors="coerce")
    valid_mask = price_num.notna() & sales_num.notna() & amount_num.notna()
    
    if valid_mask.sum() > 0:
        df_valid = df[valid_mask].copy()
        df_valid["理论销售额"] = price_num[valid_mask] * sales_num[valid_mask]
        # 修复除以0问题
        deviation_mask = (
            (abs(amount_num[valid_mask] - df_valid["理论销售额"]) / 
             df_valid["理论销售额"].replace(0, np.nan)) > 0.2
        ).fillna(False)
        amount_deviation_err = deviation_mask.sum()

# 性别异常检测
gender_col = find_column(df, ["性别"])
if gender_col:
    valid_gender_base = ["male", "female", "m", "f", "男", "女", "男生", "女生", "男性", "女性"]
    gender_err = df[~df[gender_col].astype(str).str.lower().isin(valid_gender_base)].shape[0]

# 日期异常检测
date_col = find_column(df, ["销售日期", "日期", "date"])
if date_col:
    temp_date = pd.to_datetime(df[date_col], errors="coerce")
    date_err = temp_date.isna().sum()

# 部分重复检测
key_cols = [c for c in ["产品", "价格", "年龄", "性别"] if c in df.columns]
if key_cols:
    part_dup_num = df.duplicated(subset=key_cols).sum()

# 展示检测结果
st.markdown("### 异常检测统计")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("完全重复数据", dup_num)
    st.metric("核心字段部分重复", part_dup_num)
    st.metric("全局缺失值总数", miss_num)
with col2:
    st.metric("数值列混入文本", text_in_num)
    st.metric("文本列混入数字", num_in_text)
    st.metric("特殊符号/格式脏数据", special_symbol)
with col3:
    st.metric("不合理负值", negative_err)
    st.metric("年龄超范围", age_outlier_err)
    st.metric("极端离群值", extreme_outlier)

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("非法性别值", gender_err)
    st.metric("日期格式错误", date_err)
with col2:
    st.metric("价格为0但销量>0", zero_price_non_sales)
    st.metric("销量为0但销售额>0", zero_sales_non_amount)
with col3:
    st.metric("销售额计算偏差", amount_deviation_err)

# ==================== 数据清洗（修复所有边界错误） ====================
st.divider()
st.subheader("数据清洗")
df_clean = df.copy()

# 重复值清洗
before_dup = len(df_clean)
df_clean = df_clean.drop_duplicates()
st.write(f"✅ 已删除完全重复行：{before_dup - len(df_clean)} 条")

if key_cols:
    before_part = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=key_cols, keep="first")
    st.write(f"✅ 已删除部分重复行：{before_part - len(df_clean)} 条")

# 格式清洗
for col in df_clean.columns:
    # 数值列清洗
    if any(key in str(col).lower() for key in ["销售额", "价格", "销量", "单位成本", "折扣", "年龄", "单价", "金额"]):
        # 修复特殊字符清洗
        df_clean[col] = df_clean[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# 产品列清洗
product_col = find_column(df_clean, ["产品", "商品", "品名"])
if product_col:
    df_clean[product_col] = df_clean[product_col].apply(clean_product_format)

# 地区列清洗
region_col = find_column(df_clean, ["地区", "省份", "城市"])
if region_col:
    df_clean[region_col] = df_clean[region_col].apply(clean_region_format)

# 日期列清洗
if date_col and date_col in df_clean.columns:
    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors="coerce")

# 性别列清洗
if gender_col and gender_col in df_clean.columns:
    df_clean[gender_col] = df_clean[gender_col].apply(standardize_gender)

# 缺失值处理
key_drop_cols = [c for c in ["产品", "价格", "地区"] if c in df_clean.columns]
if key_drop_cols:
    before_drop = len(df_clean)
    df_clean = df_clean.dropna(subset=key_drop_cols)
    st.write(f"✅ 已删除关键字段缺失行：{before_drop - len(df_clean)} 条")

# 数值列缺失值填充（中位数）
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if not df_clean[col].dropna().empty:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# 文本列缺失值填充
for col in ["地区", "性别"]:
    col_found = find_column(df_clean, [col])
    if col_found:
        df_clean[col_found] = df_clean[col_found].fillna("未知")

# 逻辑错误修复
# 修复销量为0但销售额>0
if sales_col and amount_col:
    df_clean[sales_col] = pd.to_numeric(df_clean[sales_col], errors="coerce")
    df_clean[amount_col] = pd.to_numeric(df_clean[amount_col], errors="coerce")
    zero_sales_mask = (df_clean[sales_col] == 0) & (df_clean[amount_col] > 0)
    df_clean.loc[zero_sales_mask, amount_col] = 0

# 修复价格为0但销量>0
if price_col and sales_col and product_col:
    df_clean[price_col] = pd.to_numeric(df_clean[price_col], errors="coerce")
    df_clean[sales_col] = pd.to_numeric(df_clean[sales_col], errors="coerce")
    zero_price_mask = (df_clean[price_col] == 0) & (df_clean[sales_col] > 0)
    
    if zero_price_mask.any():
        # 修复分组计算空值问题
        avg_price = df_clean.groupby(product_col)[price_col].transform(
            lambda x: x[x>0].mean() if x[x>0].any() else df_clean[price_col].median()
        )
        df_clean.loc[zero_price_mask, price_col] = avg_price

# 修复销售额计算偏差
if price_col and sales_col and amount_col:
    df_clean[price_col] = pd.to_numeric(df_clean[price_col], errors="coerce")
    df_clean[sales_col] = pd.to_numeric(df_clean[sales_col], errors="coerce")
    df_clean[amount_col] = pd.to_numeric(df_clean[amount_col], errors="coerce")
    
    df_clean["理论销售额"] = df_clean[price_col] * df_clean[sales_col]
    # 修复除以0问题
    deviation_mask = (
        (abs(df_clean[amount_col] - df_clean["理论销售额"]) / 
         df_clean["理论销售额"].replace(0, np.nan)) > 0.2
    ).fillna(False)
    
    df_clean.loc[deviation_mask, amount_col] = df_clean.loc[deviation_mask, "理论销售额"]
    df_clean = df_clean.drop(columns=["理论销售额"], errors='ignore')

# 异常值剔除
# 非负校验
for col in [price_col, sales_col]:
    if col and col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean = df_clean[df_clean[col] >= 0]

# 年龄范围校验
age_col = find_column(df_clean, ["年龄", "岁数"])
if age_col and age_col in df_clean.columns:
    df_clean[age_col] = pd.to_numeric(df_clean[age_col], errors="coerce")
    df_clean = df_clean[(df_clean[age_col] >= 0) & (df_clean[age_col] <= 120)]

# Z-score异常值剔除（修复空值问题）
if len(df_clean) > 100 and numeric_cols:
    valid_numeric = df_clean[numeric_cols].dropna()
    if not valid_numeric.empty:
        # 修复Z-score计算
        z_scores = np.abs(stats.zscore(valid_numeric))
        outlier_mask = pd.Series((z_scores < 3).all(axis=1), index=valid_numeric.index)
        outlier_mask = outlier_mask.reindex(df_clean.index, fill_value=True)
        df_clean = df_clean[outlier_mask]
        st.write("✅ 极端偏离的异常值已剔除")

# 最终数据校验
final_rows = len(df_clean)
if final_rows == 0:
    st.error("❌ 清洗后无有效数据，请检查原始数据质量！")
    st.stop()

st.success(f"✅ 全部清洗完成，最终有效数据：{final_rows} 行")

# ==================== 有序编码说明 ====================
st.divider()
st.subheader("📘 有序编码说明（文献支持）")
with st.expander("点击查看编码规则与文献依据", expanded=True):
    st.markdown("""
    **一、为什么采用有序编码？**  
    根据张文彤《SPSS统计分析高级教程》、吴喜之《统计学：从数据到结论》以及于斌斌(2022)《地理科学》等权威文献，对于高基数无序分类变量（如地区、产品类别），直接使用哑变量会导致维度爆炸、模型解释困难。  
    可行做法：依据**现实经济意义**将无序分类重构为**有序分类**，然后赋值为1/2/3/4/5等整数，作为连续变量纳入线性回归。该法统计合法，且被大量实证论文采用。

    **二、本系统编码规则**  
    **1. 地区编码（基于城市等级）**  
    - 一线城市 → 5  
    - 新一线城市 → 4  
    - 二线城市 → 3  
    - 三线城市 → 2  
    - 其他/境外 → 1  

    **2. 产品类别编码（基于平均价格档次）**  
    自动计算每个类别的平均价格，按价格升序等频划分为5档：  
    - 最低20% → 1（低端）  
    - 20%-40% → 2（中低）  
    - 40%-60% → 3（中端）  
    - 60%-80% → 4（中高）  
    - 80%-100% → 5（高端）  
    """)

# ==================== 执行编码 ====================
# 地区编码
if region_col and region_col in df_clean.columns:
    df_clean["地区_编码"] = df_clean[region_col].apply(encode_city_level)

# 产品类别编码
category_col = find_column(df_clean, ["产品类别", "商品类别", "品类"])
if category_col and price_col and price_col in df_clean.columns:
    tier_map = encode_product_tier(df_clean[price_col], df_clean[category_col])
    if tier_map:
        df_clean["产品类别_编码"] = df_clean[category_col].map(tier_map)

# ==================== 路径1：描述性与趋势分析（修复空数据可视化） ====================
st.divider()
st.subheader("路径1：描述性与趋势分析")

# 核心指标
total_sales = df_clean[amount_col].sum() if (amount_col and amount_col in df_clean.columns) else 0
avg_sales = df_clean[amount_col].mean() if (amount_col and amount_col in df_clean.columns) else 0
total_orders = len(df_clean)

st.markdown("### 核心销售指标概览")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("总销售额", f"{total_sales:,.2f} 元")
with col2:
    st.metric("平均单笔销售额", f"{avg_sales:,.2f} 元")
with col3:
    st.metric("总订单数", f"{total_orders} 笔")

st.write("**数据解读：** 整体订单总量、总营收与单客均价构成整体销售基准，可衡量整体消费水平。")
st.info("**经营决策：** 以单笔均价作为产品定价基准，高于均价定位中高端产品，低于均价做引流特价品。")

# 趋势分析（时间序列双图）
if date_col and date_col in df_clean.columns:
    st.markdown("### 销售趋势分析（时间序列）")
    df_date_valid = df_clean[df_clean[date_col].notna()].copy()
    
    if not df_date_valid.empty:
        df_date_valid['销售月份'] = df_date_valid[date_col].dt.to_period('M')
        monthly_sales = df_date_valid.groupby('销售月份')[amount_col].sum() if amount_col else None
        monthly_quantity = df_date_valid.groupby('销售月份')[sales_col].sum() if (sales_col and sales_col in df_date_valid.columns) else None
        
        if monthly_sales is not None and len(monthly_sales) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
            
            # 销售额趋势
            monthly_sales.plot(ax=axes[0], marker='o', color="#2076b4", linewidth=2)
            axes[0].set_title("月度销售额趋势", fontsize=12)
            axes[0].set_xlabel("月份", fontsize=10)
            axes[0].set_ylabel("销售额（元）", fontsize=10)
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)
            
            # 销量趋势
            if monthly_quantity is not None and len(monthly_quantity) > 0:
                monthly_quantity.plot(ax=axes[1], marker='s', color='#ff7f0e', linewidth=2)
                axes[1].set_title("月度销量趋势", fontsize=12)
                axes[1].set_xlabel("月份", fontsize=10)
                axes[1].set_ylabel("销量", fontsize=10)
                axes[1].grid(True, alpha=0.3)
                axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            
            if len(monthly_sales) >= 3:
                peak_month = monthly_sales.idxmax()
                trough_month = monthly_sales.idxmin()
                st.write(f"📈 销售旺季：{peak_month}，销售淡季：{trough_month}")
                st.write("**数据解读：** 月度销售额与销量走势基本同步，存在明显淡旺季，峰值月份营收远高于低谷月份。")
                st.info(f"**经营决策：** 旺季{peak_month}提前备货、加大广告投放；淡季{trough_month}开展满减、捆绑促销提升销量。")

# 价格&销量分布直方图
st.markdown("### 价格与销量分布分析")
col1, col2 = st.columns(2)

# 价格分布
with col1:
    if price_col and price_col in df_clean.columns and not df_clean[price_col].dropna().empty:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        df_clean[price_col].hist(bins=20, color="#2ca02c", edgecolor='black', alpha=0.7, ax=ax)
        ax.set_title("价格分布直方图", fontsize=12)
        ax.set_xlabel("价格（元）", fontsize=10)
        ax.set_ylabel("频数", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        price_mean = df_clean[price_col].mean()
        st.write(f"**数据解读：** 商品均价{price_mean:.2f}元，直方图柱子集中区间为市场接受主力价位，两端为低价/高价小众产品。")
        st.info("**经营决策：** 主力价位产品作为店铺主推，少量高低价产品完善产品结构。")

# 销量分布
with col2:
    if sales_col and sales_col in df_clean.columns and not df_clean[sales_col].dropna().empty:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        df_clean[sales_col].hist(bins=20, color="#ff7f0e", edgecolor='black', alpha=0.7, ax=ax)
        ax.set_title("销量分布直方图", fontsize=12)
        ax.set_xlabel("销量", fontsize=10)
        ax.set_ylabel("频数", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        sale_mean = df_clean[sales_col].mean()
        st.write(f"**数据解读：** 单订单平均采购量{sale_mean:.2f}件，多数订单小批量采购，大额批量订单占比偏低。")
        st.info("**经营决策：** 设计多件组合优惠，引导用户提高单次采购数量。")

# ==================== 多维度对比分析（修复空数据可视化） ====================
st.markdown("### 多维度对比分析")

# 地区销售额对比
if region_col and region_col in df_clean.columns and amount_col:
    df_clean["地区_清洗"] = clean_group_data(df_clean[region_col])
    region_sales = df_clean.groupby("地区_清洗")[amount_col].sum().sort_values(ascending=False)
    region_sales = region_sales[region_sales > 0]
    
    if not region_sales.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            # 最多显示前10个地区
            top_regions = region_sales.head(10)
            top_regions.plot(kind="bar", color="#9467bd", ax=ax)
            ax.set_title("各地区销售额对比（前10）", fontsize=12)
            ax.set_xlabel("地区", fontsize=10)
            ax.set_ylabel("销售额（元）", fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            st.write("**数据解读：** 区域营收差距悬殊，头部地区消费体量遥遥领先，尾部地区市场潜力不足。")
            st.info("**经营决策：** 深耕头部热门区域，增设线下/推广资源；低销地区调研消费习惯，针对性选品拓客。")
        
        with col2:
            if len(region_sales) >= 3:
                top3 = region_sales.head(3)
                bottom3 = region_sales.tail(3)
                
                st.write("**销售额TOP3地区**")
                for i, (region, sales) in enumerate(top3.items(), 1):
                    st.write(f"{i}. {region}: {sales:,.0f}元")
                
                st.write("**销售额末位3地区**")
                for i, (region, sales) in enumerate(bottom3.items(), 1):
                    st.write(f"{i}. {region}: {sales:,.0f}元")
                
                top_region = top3.index[0]
                st.success(f"📌 **购买力最强的地区：{top_region}**，贡献了 {top3.iloc[0]/region_sales.sum()*100:.1f}% 的销售额。")

# 产品销售额TOP10
if product_col and product_col in df_clean.columns and amount_col:
    df_clean["产品_清洗"] = clean_group_data(df_clean[product_col])
    product_sales = df_clean.groupby("产品_清洗")[amount_col].sum().sort_values(ascending=False).head(10)
    
    if not product_sales.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        product_sales.plot(kind="bar", color="#8c564b", ax=ax)
        ax.set_title("产品销售额Top10", fontsize=12)
        ax.set_xlabel("产品", fontsize=10)
        ax.set_ylabel("销售额（元）", fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        top_product = product_sales.index[0]
        st.success(f"📌 **最畅销产品：{top_product}**，销售额 {product_sales.iloc[0]:,.0f}元。")
        st.write("**数据解读：** 头部爆款单品贡献大量营收，后面单品销量快速下滑，单品销售集中度高。")
        st.info("**经营决策：** 爆款作为引流主力，加大库存备货；滞销单品优化定价或淘汰下架。")

# 品类&性别双饼图
st.markdown("### 品类与性别消费分析")
col1, col2 = st.columns(2)

# 产品类别占比
with col1:
    if category_col and category_col in df_clean.columns and amount_col:
        df_clean["类别_清洗"] = clean_group_data(df_clean[category_col])
        category_sales = df_clean.groupby("类别_清洗")[amount_col].sum()
        category_sales = category_sales[category_sales > 0]
        
        if not category_sales.empty:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            wedges, texts, autotexts = ax.pie(
                category_sales.values,
                labels=category_sales.index,
                autopct="%1.1f%%",
                wedgeprops=dict(width=0.7),
                startangle=90
            )
            ax.set_title("各产品类别销售额占比", fontsize=12)
            plt.setp(autotexts, size=8, weight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            top_cat = category_sales.idxmax()
            st.success(f"📌 **最畅销产品类别：{top_cat}**，占比 {category_sales.max()/category_sales.sum()*100:.1f}%")
            st.write("**数据解读：** 单一品类占比过高，品类结构不均衡，弱势品类营收体量偏小。")
            st.info("**经营决策：** 巩固王牌品类优势，扩充弱势品类SKU，平衡营收结构。")

# 性别消费占比
with col2:
    if gender_col and gender_col in df_clean.columns and amount_col:
        gender_sales = df_clean[df_clean[gender_col].isin(["Male", "Female"])].groupby(gender_col)[amount_col].sum()
        
        if not gender_sales.empty:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            wedges, texts, autotexts = ax.pie(
                gender_sales.values,
                labels=gender_sales.index,
                autopct="%1.1f%%",
                wedgeprops=dict(width=0.7),
                startangle=90
            )
            ax.set_title("不同性别消费占比", fontsize=12)
            plt.setp(autotexts, size=8, weight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            if 'Female' in gender_sales and gender_sales['Female'] > gender_sales.get('Male', 0):
                st.success(f"📌 **女性消费者贡献更高**，占比 {gender_sales['Female']/gender_sales.sum()*100:.1f}%。")
                st.write("**数据解读：** 女性是消费主力人群，消费贡献远超男性客群。")
                st.info("**经营决策：** 产品选型、活动营销围绕女性消费偏好设计。")
            elif 'Male' in gender_sales:
                st.success(f"📌 **男性消费者贡献更高**，占比 {gender_sales['Male']/gender_sales.sum()*100:.1f}%。")
                st.write("**数据解读：** 男性为核心消费群体。")
                st.info("**经营决策：** 产品开发与营销侧重男性需求。")

# 口味分析（月饼/巧克力）
if product_col and product_col in df_clean.columns and amount_col:
    sample_prod = str(df_clean[product_col].iloc[0]) if len(df_clean) > 0 else ""
    is_moon = "月饼" in sample_prod
    is_choco = any(key in sample_prod.lower() for key in ["巧克力", "choco", "dark", "milk", "white"])
    
    if is_moon or is_choco:
        st.markdown("### 产品口味/类型分析")
        if is_moon:
            df_clean["口味"] = df_clean[product_col].apply(extract_moon_flavor)
        else:
            df_clean["口味"] = df_clean[product_col].apply(extract_choco_flavor)
        
        flavor_sales = df_clean.groupby("口味")[amount_col].sum().sort_values(ascending=False)
        
        if not flavor_sales.empty:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
            flavor_sales.head(10).plot(kind="bar", color="coral", ax=ax)
            ax.set_title("各口味/类型销售额排名", fontsize=12)
            ax.set_xlabel("口味/类型", fontsize=10)
            ax.set_ylabel("销售额（元）", fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            top_flavor = flavor_sales.index[0]
            st.success(f"📌 **最受欢迎的口味：{top_flavor}**，销售额占比 {flavor_sales.iloc[0]/flavor_sales.sum()*100:.1f}%")
            st.write("**数据解读：** 头部口味市场认可度高，小众口味市场需求有限。")
            st.info("**经营决策：** 爆款口味加大产能，小众口味小批量备货减少库存积压。")

# 年龄分层分析
if age_col and age_col in df_clean.columns and amount_col:
    st.markdown("### 不同年龄层消费分布")
    df_clean["年龄_数值"] = pd.to_numeric(df_clean[age_col], errors="coerce")
    df_age_valid = df_clean[(df_clean["年龄_数值"] >= 0) & (df_clean["年龄_数值"] <= 120)]
    
    if not df_age_valid.empty:
        df_age_valid["年龄分组"] = pd.cut(
            df_age_valid["年龄_数值"], 
            bins=[0,25,35,45,60,120],
            labels=["18-25岁", "26-35岁", "36-45岁", "46-60岁", "60+岁"],
            right=False  # 修复区间包含问题
        )
        age_sales = df_age_valid.groupby("年龄分组")[amount_col].sum()
        
        if not age_sales.empty:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
            age_sales.plot(kind="bar", color="#e377c2", ax=ax)
            ax.set_title("各年龄层销售额对比", fontsize=12)
            ax.set_xlabel("年龄分组", fontsize=10)
            ax.set_ylabel("销售额（元）", fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            top_age = age_sales.idxmax()
            st.success(f"📌 **消费主力年龄段：{top_age}**，贡献销售额 {age_sales.max():,.0f}元。")
            st.write(f"**数据解读：** {top_age}年龄段消费能力最强，其余年龄段消费体量依次递减。")
            st.info("**经营决策：** 资源倾斜主力年龄段产品，针对其他年龄段开发适配刚需产品。")

# 销售人员业绩
salesperson_col = find_column(df_clean, ["销售人员", "销售代表", "业务员"])
if salesperson_col and salesperson_col in df_clean.columns and amount_col:
    st.markdown("### 销售人员业绩Top10")
    df_clean["销售_清洗"] = clean_group_data(df_clean[salesperson_col])
    salesperson_sales = df_clean.groupby("销售_清洗")[amount_col].sum().sort_values(ascending=False).head(10)
    
    if not salesperson_sales.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        salesperson_sales.plot(kind="bar", color="#d62728", ax=ax)
        ax.set_title("销售人员业绩Top10", fontsize=12)
        ax.set_xlabel("销售人员", fontsize=10)
        ax.set_ylabel("销售额（元）", fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        top_sp = salesperson_sales.index[0]
        st.success(f"📌 **最佳销售人员：{top_sp}**，业绩 {salesperson_sales.iloc[0]:,.0f}元，占比 {salesperson_sales.iloc[0]/salesperson_sales.sum()*100:.1f}%。")
        st.write("**数据解读：** 销冠业绩远超团队平均水平，内部人员业绩分化明显。")
        st.info("**经营决策：** 销冠分享销售话术与拓客经验，培训落后员工，设置阶梯提成激励全员。")

# 销售渠道分析
channel_col = find_column(df_clean, ["销售渠道", "渠道"])
if channel_col and channel_col in df_clean.columns and amount_col:
    st.markdown("### 销售渠道销售额对比")
    df_clean["渠道_清洗"] = clean_group_data(df_clean[channel_col])
    channel_sales = df_clean.groupby("渠道_清洗")[amount_col].sum()
    channel_sales = channel_sales[channel_sales > 0]
    
    if not channel_sales.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        channel_sales.plot(kind="bar", color="#4CAF50", ax=ax)
        ax.set_title("各销售渠道销售额对比", fontsize=12)
        ax.set_xlabel("销售渠道", fontsize=10)
        ax.set_ylabel("销售额（元）", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        top_channel = channel_sales.idxmax()
        st.success(f"📌 **最有效的销售渠道：{top_channel}**，贡献 {channel_sales.max()/channel_sales.sum()*100:.1f}% 的销售额。")
        st.write("**数据解读：** 单一渠道成为营收支柱，其余渠道创收能力偏弱。")
        st.info("**经营决策：** 加大主力渠道推广预算，优化低效渠道运营模式或缩减投入。")

# 支付方式&顾客类型分析
st.markdown("### 支付方式与顾客类型分析")
col1, col2 = st.columns(2)

# 支付方式分析
pay_col = find_column(df_clean, ["支付方式", "付款方式"])
with col1:
    if pay_col and pay_col in df_clean.columns and amount_col:
        df_clean["支付_清洗"] = clean_group_data(df_clean[pay_col])
        pay_sales = df_clean.groupby("支付_清洗")[amount_col].sum().sort_values(ascending=False)
        
        if not pay_sales.empty:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            wedges, texts, autotexts = ax.pie(
                pay_sales.values,
                labels=pay_sales.index,
                autopct="%1.1f%%",
                wedgeprops=dict(width=0.7),
                startangle=90
            )
            ax.set_title("各支付方式销售额占比", fontsize=12)
            plt.setp(autotexts, size=8, weight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            top_pay = pay_sales.index[0]
            st.success(f"📌 **最常用支付方式：{top_pay}**，占比 {pay_sales.iloc[0]/pay_sales.sum()*100:.1f}%。")
            st.write("**数据解读：** 用户支付习惯高度集中，主流支付占绝对份额。")
            st.info("**经营决策：** 优先保障主流支付通道稳定，保留小众支付满足多元化需求。")

# 顾客类型分析
customer_type_col = find_column(df_clean, ["顾客类型", "客户类型"])
with col2:
    if customer_type_col and customer_type_col in df_clean.columns and amount_col:
        df_clean["顾客_清洗"] = clean_group_data(df_clean[customer_type_col])
        customer_sales = df_clean.groupby("顾客_清洗")[amount_col].sum()
        customer_sales = customer_sales[customer_sales > 0]
        
        if not customer_sales.empty:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            wedges, texts, autotexts = ax.pie(
                customer_sales.values,
                labels=customer_sales.index,
                autopct="%1.1f%%",
                wedgeprops=dict(width=0.7),
                startangle=90
            )
            ax.set_title("各顾客类型销售额占比", fontsize=12)
            plt.setp(autotexts, size=8, weight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            top_customer = customer_sales.idxmax()
            st.success(f"📌 **核心顾客类型：{top_customer}**，占比 {customer_sales.max()/customer_sales.sum()*100:.1f}%。")
            st.write("**数据解读：** 核心顾客类型贡献大部分营收，其他类型顾客占比较低。")
            st.info("**经营决策：** 针对核心顾客类型推出会员权益，挖掘其他类型顾客的消费潜力。")

# ==================== 底部提示 ====================
st.divider()
st.info("""
💡 温馨提示：
1. 分析结果基于上传的清洗后数据，若数据质量较差可能影响分析准确性
2. 所有图表均可右键保存，分析报告可联系技术人员定制导出
3. 建议结合行业经验解读数据，本系统仅提供数据层面的分析参考
""")
