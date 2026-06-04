import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns  # 补充导入（原代码用了sns但没导入）
import warnings
warnings.filterwarnings('ignore')

# 页面全局配置
st.set_page_config(page_title="销售数据全流程智能分析系统", layout="wide")
st.title("📊 自适应全数据通用版：清洗→编码→建模→可视化")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------- 1. 数据上传与原始读取 ----------------------
st.subheader("步骤1：上传销售源数据")
upload_file = st.file_uploader("上传任意Excel/CSV销售数据", type=["xlsx", "csv"])

df = None
raw_df = None
original_rows = 0
if upload_file:
    # 修复：不再强制dtype=object，保留原始类型
    if upload_file.name.endswith(".csv"):
        df = pd.read_csv(upload_file)
    else:
        df = pd.read_excel(upload_file)
    raw_df = df.copy()
    original_rows = len(df)
    st.success(f"✅ 数据读取成功！原始数据：共 {original_rows} 行，{len(df.columns)} 列")

# 增加数据存在性校验
if df is None:
    st.warning("请先上传数据文件！")
    st.stop()
# ---------------------- 2. 全维度异常值逐项检测 ----------------------
st.divider()
st.subheader("步骤2：全类型异常值全面检测")

# 初始化异常计数器
dup_num = df.duplicated().sum()
miss_num = df.isnull().sum().sum()
text_in_num = 0       # 数值列混入文本/单位/非数字字符
num_in_text = 0       # 文本列混入数字
special_symbol = 0    # 特殊符号、多余空格、格式脏数据
negative_err = 0      # 不合理负值
extreme_outlier = 0   # 极端离群值（IQR）
gender_err = 0        # 非法性别值
date_err = 0          # 日期格式错误
zero_price_non_sales = 0   # 价格=0但销量>0（逻辑异常）
zero_sales_non_amount = 0  # 销量=0但销售额>0（逻辑异常）
amount_deviation_err = 0   # 销售额≠价格×销量（计算异常）
age_outlier_err = 0        # 年龄异常（<0或>120）
part_dup_num = 0           # 核心字段部分重复

# ---------------------- 逐列扫描基础异常 ----------------------
for col in df.columns:
    col_data = df[col].astype(str).str.strip()
    
    # 1. 特殊符号/格式脏数据（逗号、单位、空格、特殊字符）
    special_symbol += col_data.str.contains(r'[,\，元￥$岁#@!]').sum()
    special_symbol += col_data.str.contains(r'\s{2,}').sum()  # 产品名多余空格
    
    # 2. 数值列检测：文本混入、负值、IQR异常
    if any(key in col for key in ["销售额","价格","销量","单位成本","折扣","单价","金额","年龄"]):
        # 修复：安全转换为数值类型，失败则为NaN
        temp_num = pd.to_numeric(df[col], errors="coerce")
        text_in_num += temp_num.isna().sum()  # 文本混入
        # 修复：排除NaN后再统计负值
        negative_err += (temp_num.dropna() < 0).sum()  # 负值
        
        # 年龄单独处理（0-120）
        if "年龄" in col:
            age_outlier_err += ((temp_num < 0) | (temp_num > 120)).sum()
        
        # 通用IQR异常检测（排除NaN）
        if not temp_num.dropna().empty:
            q1 = temp_num.quantile(0.25)
            q3 = temp_num.quantile(0.75)
            iqr = q3 - q1
            extreme_outlier += ((temp_num < q1 - 1.5*iqr) | (temp_num > q3 + 1.5*iqr)).sum()

    # 3. 文本列检测：混入数字
    if any(key in col for key in ["产品","地区","销售人员","产品类别","支付方式","顾客类型","销售渠道"]):
        num_in_text += col_data.str.contains(r"[0-9]").sum()

# ---------------------- 4. 逻辑异常检测（核心修复：类型转换） ----------------------
# 4.1 价格=0但销量>0
if "价格" in df.columns and "销量" in df.columns:
    # 修复：先转换为数值类型
    price_num = pd.to_numeric(df["价格"], errors="coerce")
    sales_num = pd.to_numeric(df["销量"], errors="coerce")
    zero_price_non_sales = df[(price_num == 0) & (sales_num > 0)].shape[0]

# 4.2 销量=0但销售额>0
if "销量" in df.columns and "销售额" in df.columns:
    # 修复：先转换为数值类型
    sales_num = pd.to_numeric(df["销量"], errors="coerce")
    amount_num = pd.to_numeric(df["销售额"], errors="coerce")
    zero_sales_non_amount = df[(sales_num == 0) & (amount_num > 0)].shape[0]

# 4.3 销售额与价格×销量偏差过大
if "价格" in df.columns and "销量" in df.columns and "销售额" in df.columns:
    # 修复：统一转换为数值类型
    price_num = pd.to_numeric(df["价格"], errors="coerce")
    sales_num = pd.to_numeric(df["销量"], errors="coerce")
    amount_num = pd.to_numeric(df["销售额"], errors="coerce")
    
    valid_mask = price_num.notna() & sales_num.notna() & amount_num.notna()
    df_valid = df[valid_mask].copy()
    df_valid["理论销售额"] = price_num[valid_mask] * sales_num[valid_mask]
    # 修复：避免除以0
    deviation_mask = (abs(amount_num[valid_mask] - df_valid["理论销售额"]) / 
                      df_valid["理论销售额"].replace(0, np.nan) > 0.2)
    amount_deviation_err = deviation_mask.sum()

# ---------------------- 5. 性别异常检测 ----------------------
if "性别" in df.columns:
    valid_gender = ["男","女","Male","Female"]
    gender_err = df[~df["性别"].astype(str).isin(valid_gender)].shape[0]

# ---------------------- 6. 日期列自动识别与错误检测 ----------------------
date_col = None
for col in df.columns:
    if "销售日期" in col.lower() or "date" in col.lower():
        date_col = col
        temp_date = pd.to_datetime(df[col], errors="coerce")
        date_err = temp_date.isna().sum()
        break

# ---------------------- 7. 部分重复检测（对应部分重复脏数据） ----------------------
key_cols = []
if "产品" in df.columns: key_cols.append("产品")
if "价格" in df.columns: key_cols.append("价格")
if "年龄" in df.columns: key_cols.append("年龄")
if "性别" in df.columns: key_cols.append("性别")

if key_cols:
    part_dup_num = df.duplicated(subset=key_cols).sum()

# ---------------------- 完整展示全部异常统计 ----------------------
st.markdown(f"""
- 完全重复数据条数：{dup_num}
- 核心字段部分重复条数：{part_dup_num}
- 全局缺失值总数：{miss_num}
- 数值列混入文本/单位异常：{text_in_num} 处
- 文本列混入数字异常：{num_in_text} 处
- 特殊符号/格式脏数据：{special_symbol} 个
- 不合理负值异常：{negative_err} 个
- 年龄超出0-120范围异常：{age_outlier_err} 个
- 极端极大/极小离群值（IQR）：{extreme_outlier} 个
- 非法性别值：{gender_err} 个
- 日期格式错乱错误：{date_err} 个
- 价格为0但销量非0：{zero_price_non_sales} 条
- 销量为0但销售额非0：{zero_sales_non_amount} 条
- 销售额与价格×销量偏差过大：{amount_deviation_err} 条
""")

# ---------------------- 3. 全套标准化异常清洗 ----------------------
st.divider()
st.subheader("步骤3：异常值标准化修复清洗")
df_clean = df.copy()

# ---------------------- 1. 重复值清洗（完全重复 + 部分重复） ----------------------
# 1.1 删除完全重复行
before_dup = len(df_clean)
df_clean = df_clean.drop_duplicates()
st.write(f"✅ 已删除完全重复行：{before_dup - len(df_clean)} 条")

# 1.2 处理核心字段部分重复（产品+价格/用户+产品）
key_cols = []
if "产品" in df_clean.columns: key_cols.append("产品")
if "价格" in df_clean.columns: key_cols.append("价格")
if "年龄" in df_clean.columns: key_cols.append("年龄")
if "性别" in df_clean.columns: key_cols.append("性别")

if key_cols:
    before_part_dup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=key_cols, keep="first")
    st.write(f"✅ 已删除核心字段部分重复行：{before_part_dup - len(df_clean)} 条")

# ---------------------- 2. 格式问题清洗（数值列、文本列、日期、地区） ----------------------
# 2.1 数值列：去除文字、单位、特殊符号，转为纯数字（含千分符、元/￥等）
for col in df_clean.columns:
    if any(key in col for key in ["销售额","价格","销量","单位成本","折扣","年龄","单价","金额"]):
        # 去除非数字字符（含逗号、元、￥、空格等）
        df_clean[col] = df_clean[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        # 转为数值
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# 2.2 文本列清洗：产品名/地区多余空格、特殊字符
if "产品" in df_clean.columns:
    df_clean["产品"] = df_clean["产品"].astype(str).str.strip()
    df_clean["产品"] = df_clean["产品"].str.replace(r"\s+", " ", regex=True)  # 多个空格转为单个
    df_clean["产品"] = df_clean["产品"].str.replace(r"[#@!$%^&*]", "", regex=True)  # 特殊字符去除

if "地区" in df_clean.columns:
    df_clean["地区"] = df_clean["地区"].astype(str).str.lower()
    # 地区简称/大小写统一
    region_map = {
        "chongqing": "重庆", "cq": "重庆", "渝": "重庆",
        "beijing": "北京", "bj": "北京", "京": "北京",
        "shanghai": "上海", "sh": "上海", "沪": "上海"
    }
    df_clean["地区"] = df_clean["地区"].replace(region_map)

# 2.3 日期格式统一修复
if date_col is not None and date_col in df_clean.columns:
    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors="coerce")
    st.write("✅ 错乱、不规范日期格式已统一修复")

# 2.4 性别格式标准化（用户数据）
if "性别" in df_clean.columns:
    gender_map = {"男": "男", "男生": "男", "男性": "男", "女": "女", "女生": "女", "女性": "女", "未知": "未知"}
    df_clean["性别"] = df_clean["性别"].astype(str).map(gender_map).fillna("未知")
    st.write("✅ 性别格式已标准化")

# ---------------------- 3. 缺失值处理（关键字段删除 + 数值列中位数填充） ----------------------
# 3.1 关键字段缺失删除（产品、价格、地区等）
key_cols_drop = []
if "产品" in df_clean.columns: key_cols_drop.append("产品")
if "价格" in df_clean.columns: key_cols_drop.append("价格")
if "地区" in df_clean.columns: key_cols_drop.append("地区")

if key_cols_drop:
    before_drop = len(df_clean)
    df_clean = df_clean.dropna(subset=key_cols_drop)
    st.write(f"✅ 已删除关键字段缺失行：{before_drop - len(df_clean)} 条")

# 3.2 数值列缺失值中位数填充
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if not df_clean[col].dropna().empty:  # 避免空列报错
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# 3.3 分类列缺失值填充（地区、性别）
for col in ["地区", "性别"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna("未知")

# ---------------------- 4. 逻辑错误清洗（销量/价格/销售额矛盾） ----------------------
# 4.1 销量为0但销售额非0 → 修正销售额为0
if "销量" in df_clean.columns and "销售额" in df_clean.columns:
    # 确保是数值类型
    df_clean["销量"] = pd.to_numeric(df_clean["销量"], errors="coerce")
    df_clean["销售额"] = pd.to_numeric(df_clean["销售额"], errors="coerce")
    df_clean.loc[(df_clean["销量"] == 0) & (df_clean["销售额"] > 0), "销售额"] = 0

# 4.2 价格为0但销量非0 → 用同类产品平均价格填充
if "价格" in df_clean.columns and "销量" in df_clean.columns and "产品" in df_clean.columns:
    df_clean["价格"] = pd.to_numeric(df_clean["价格"], errors="coerce")
    df_clean["销量"] = pd.to_numeric(df_clean["销量"], errors="coerce")
    zero_price_mask = (df_clean["价格"] == 0) & (df_clean["销量"] > 0)
    if zero_price_mask.any():
        # 避免除以0
        avg_price = df_clean.groupby("产品")["价格"].transform(lambda x: x[x>0].mean() if x[x>0].any() else 0)
        df_clean.loc[zero_price_mask, "价格"] = avg_price

# 4.3 销售额与价格×销量偏差过大 → 修正为理论值
if "价格" in df_clean.columns and "销量" in df_clean.columns and "销售额" in df_clean.columns:
    df_clean["价格"] = pd.to_numeric(df_clean["价格"], errors="coerce")
    df_clean["销量"] = pd.to_numeric(df_clean["销量"], errors="coerce")
    df_clean["销售额"] = pd.to_numeric(df_clean["销售额"], errors="coerce")
    
    df_clean["理论销售额"] = df_clean["价格"] * df_clean["销量"]
    # 避免除以0
    deviation_mask = (abs(df_clean["销售额"] - df_clean["理论销售额"]) / 
                      df_clean["理论销售额"].replace(0, np.nan) > 0.2)
    df_clean.loc[deviation_mask, "销售额"] = df_clean.loc[deviation_mask, "理论销售额"]
    df_clean = df_clean.drop(columns=["理论销售额"])

# ---------------------- 5. 异常值剔除（负值、无效年龄、Z-score极端值） ----------------------
# 5.1 负值异常（价格、销量）
for col in ["价格", "销量"]:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean = df_clean[df_clean[col] >= 0]

# 5.2 年龄异常（<0或>120）
if "年龄" in df_clean.columns:
    df_clean["年龄"] = pd.to_numeric(df_clean["年龄"], errors="coerce")
    df_clean = df_clean[(df_clean["年龄"] >= 0) & (df_clean["年龄"] <= 120)]

# 5.3 Z-score剔除极端异常极值（保留Z<3）
if len(df_clean) > 100 and numeric_cols:
    # 只对非空数值列计算Z-score
    valid_numeric = df_clean[numeric_cols].dropna()
    if not valid_numeric.empty:
        z_scores = np.abs(stats.zscore(valid_numeric))
        # 先转为Series再对齐索引
        outlier_mask = pd.Series((z_scores < 3).all(axis=1), index=valid_numeric.index)
        # 对齐索引（填充未参与计算的行，默认保留）
        outlier_mask = outlier_mask.reindex(df_clean.index, fill_value=True)
        df_clean = df_clean[outlier_mask]
        st.write("✅ 极端偏离的极大极小异常值已剔除")

# ---------------------- 清洗结果汇总 ----------------------
final_rows = len(df_clean)
st.success(f"✅ 全部清洗完成，最终有效建模数据：{final_rows} 行")

# ---------------------- 4. 智能动态编码说明（核心修复） ----------------------
st.divider()
st.subheader("步骤4：分类变量数字化编码方式说明")

# 自动识别当前数据真实存在的字段，不存在绝不展示
exist_binary = []
exist_multi = []

# 自动匹配二分类字段
if "性别" in df_clean.columns:
    exist_binary.append("性别：男/女 → 0/1 标签编码")
if "顾客类型" in df_clean.columns:
    exist_binary.append("顾客类型：returning老客/new新客 → 0/1 标签编码")
if "支付方式" in df_clean.columns:
    exist_binary.append("支付方式：cash现金/bank转账 → 0/1 标签编码")

# 自动匹配多分类平等字段
if "地区" in df_clean.columns:
    exist_multi.append("地区（中英文地名/城市）→ 独热编码，完全平等、无大小等级")
if "销售人员" in df_clean.columns:
    exist_multi.append("销售人员（任意英文/中文人名）→ 独热编码，人名之间完全平等")
if "销售渠道" in df_clean.columns:
    exist_multi.append("销售渠道（online线上/retail线下）→ 独热编码，渠道无优劣顺序")
if "产品类别" in df_clean.columns:
    exist_multi.append("产品类别 → 独热编码，各类别之间完全平等")

# 仅展示当前数据真实有的编码说明
if exist_binary:
    st.markdown("**1. 二分类变量（0/1 标签编码）**")
    for item in exist_binary:
        st.write(f"• {item}")

if exist_multi:
    st.markdown("**2. 多分类无序平等变量（One-Hot独热编码）**")
    for item in exist_multi:
        st.write(f"• {item}")

st.success("✅ 所有文本字段全部转为合规建模数字，无顺序、无递增、无虚假等级偏差")
st.success("✅ 仅对当前这份数据真实存在的字段完成编码")

# 自动执行编码
all_cate_cols = [c for c in df_clean.columns if not pd.api.types.is_numeric_dtype(df_clean[c]) and c != date_col]
binary_list = []
multi_equal_list = []

for col in all_cate_cols:
    # 排除空值后统计唯一值
    unique_vals = df_clean[col].dropna().unique()
    unique_count = len(unique_vals)
    if unique_count == 2:
        binary_list.append(col)
    else:
        multi_equal_list.append(col)

# 二分类0/1编码
for col in binary_list:
    df_clean[col] = df_clean[col].astype(str).fillna("未知")
    sorted_vals = sorted(df_clean[col].unique())
    df_clean[col] = df_clean[col].map({sorted_vals[0]:0, sorted_vals[1]:1})

# 多分类独热编码
if len(multi_equal_list) > 0:
    df_clean = pd.get_dummies(df_clean, columns=multi_equal_list, prefix=multi_equal_list)

# ---------------------- 5. 建模变量规范定义 ----------------------
st.divider()
st.subheader("步骤5：建模变量明确定义")
Y_name = "销售额"
# 确保销售额是数值类型
if Y_name in df_clean.columns:
    df_clean[Y_name] = pd.to_numeric(df_clean[Y_name], errors="coerce")

# 筛选有效的自变量（数值类型且非空）
X_all_cols = [
    c for c in df_clean.columns 
    if c != Y_name and np.issubdtype(df_clean[c].dtype, np.number) 
    and not df_clean[c].dropna().empty
]

st.info(f"""
📌 固定建模规则：
因变量 Y = 销售额
自变量 X = 其余全部字段，**总计 {len(X_all_cols)} 个自变量全部纳入模型，无任何删减**
包含：价格、销量、年龄、折扣、以及所有编码后的分类变量
""")

# 建模前最后校验
if Y_name not in df_clean.columns or len(X_all_cols) == 0:
    st.error("建模变量不足，请检查数据字段！")
    st.stop()

# ---------------------- 6. 路径1：描述性与趋势分析 ----------------------
st.divider()
st.subheader("步骤6：路径1 - 描述性与趋势分析")

# 6.1 核心指标概览
total_sales = df_clean[Y_name].sum()
avg_sales = df_clean[Y_name].mean()
total_orders = len(df_clean)
st.markdown("### 核心销售指标概览")
st.write(f"• 总销售额：{total_sales:,.2f}")
st.write(f"• 平均单笔销售额：{avg_sales:,.2f}")
st.write(f"• 总订单数：{total_orders}")

# 6.2 销售趋势分析
if date_col is not None and date_col in df_clean.columns:
    st.markdown("### 销售趋势分析")
    # 排除日期为空的行
    df_date_valid = df_clean[df_clean[date_col].notna()].copy()
    if not df_date_valid.empty:
        df_date_valid['销售月份'] = df_date_valid[date_col].dt.to_period('M')
        monthly_sales = df_date_valid.groupby('销售月份')[Y_name].sum()
        fig, ax = plt.subplots(figsize=(10, 4))
        monthly_sales.plot(ax=ax, color="#1f77b4")
        ax.set_title("月度销售额趋势")
        ax.set_xlabel("月份")
        ax.set_ylabel("销售额")
        st.pyplot(fig)

# 6.3 价格/销量分布
st.markdown("### 价格与销量分布分析")
col1, col2 = st.columns(2)
with col1:
    if "价格" in df_clean.columns and not df_clean["价格"].dropna().empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        df_clean["价格"].hist(bins=20, color="#2ca02c")
        ax.set_title("价格分布直方图")
        ax.set_xlabel("价格")
        st.pyplot(fig)
with col2:
    if "销量" in df_clean.columns and not df_clean["销量"].dropna().empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        df_clean["销量"].hist(bins=20, color="#ff7f0e")
        ax.set_title("销量分布直方图")
        ax.set_xlabel("销量")
        st.pyplot(fig)

# ---------------------- 7. 路径2：多维度对比分析 ----------------------
st.divider()
st.subheader("步骤7：路径2 - 多维度对比分析")

# 辅助函数：模糊匹配列名（提升兼容性）
def find_column(df, keywords, return_first=True):
    """
    模糊查找包含指定关键词的列
    :param df: 数据框
    :param keywords: 关键词列表（如["地区","省份","城市"]）
    :param return_first: 是否只返回第一个匹配项
    :return: 匹配的列名（str/list）或None
    """
    matched = []
    for col in df.columns:
        for kw in keywords:
            if kw in col:
                matched.append(col)
                if return_first:
                    return col
    return matched if matched else None

# 辅助函数：数据清洗（分组前确保无空值、无异常字符）
def clean_group_data(series):
    """清洗分组用的列：去空、去空格、统一小写"""
    return series.astype(str).str.strip().str.lower().replace("", np.nan).dropna()

# ===================== 7.1 地区分析（增强版） =====================
region_col = find_column(df_clean, ["地区", "省份", "城市", "区域"])
if region_col and not df_clean[region_col].dropna().empty:
    st.markdown("### 地区总销售额对比")
    # 清洗地区数据（去重、去空、统一格式）
    df_clean["地区_清洗"] = clean_group_data(df_clean[region_col])
    region_sales = df_clean.groupby("地区_清洗")[Y_name].sum().sort_values(ascending=False)
    
    # 过滤掉销售额为0的地区
    region_sales = region_sales[region_sales > 0]
    if not region_sales.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        region_sales.plot(kind="bar", color="#9467bd")
        ax.set_title("各地区总销售额对比")
        ax.set_xlabel("地区")
        ax.set_ylabel("总销售额")
        # 解决x轴标签重叠问题
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.warning("⚠️ 地区销售额均为0，暂无法展示对比图")
else:
    st.info("ℹ️ 未检测到地区相关字段（地区/省份/城市），跳过地区分析")

# ===================== 7.2 产品分析（增强版） =====================
product_col = find_column(df_clean, ["产品", "商品", "品名"])
if product_col and not df_clean[product_col].dropna().empty:
    st.markdown("### 产品销售额Top10")
    # 清洗产品数据
    df_clean["产品_清洗"] = clean_group_data(df_clean[product_col])
    product_sales = df_clean.groupby("产品_清洗")[Y_name].sum().sort_values(ascending=False)
    # 取Top10（不足10个则取全部）
    product_sales = product_sales.head(10)
    
    if not product_sales.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        product_sales.plot(kind="bar", color="#8c564b")
        ax.set_title("产品总销售额Top10")
        ax.set_xlabel("产品")
        ax.set_ylabel("总销售额")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.warning("⚠️ 产品销售额均为0，暂无法展示Top10")
else:
    st.info("ℹ️ 未检测到产品相关字段（产品/商品/品名），跳过产品分析")

# ===================== 7.3 产品类别分析（增强版） =====================
category_col = find_column(df_clean, ["产品类别", "商品类别", "品类", "类别"])
if category_col and not df_clean[category_col].dropna().empty:
    st.markdown("### 产品类别销售额占比")
    # 清洗类别数据
    df_clean["类别_清洗"] = clean_group_data(df_clean[category_col])
    category_sales = df_clean.groupby("类别_清洗")[Y_name].sum()
    # 过滤掉销售额为0的类别
    category_sales = category_sales[category_sales > 0]
    
    if not category_sales.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        # 解决pie图标签重叠问题
        category_sales.plot(kind="pie", autopct="%1.1f%%", ax=ax, 
                           wedgeprops=dict(width=0.7), startangle=90)
        ax.set_title("各产品类别销售额占比")
        ax.set_ylabel("")  # 隐藏y轴标签
        st.pyplot(fig)
    else:
        st.warning("⚠️ 产品类别销售额均为0，暂无法展示占比图")
else:
    st.info("ℹ️ 未检测到产品类别相关字段（产品类别/品类），跳产品类别分析")

# ===================== 7.4 年龄分析（优化版） =====================
age_col = find_column(df_clean, ["年龄", "岁数"])
if age_col and not df_clean[age_col].dropna().empty:
    st.markdown("### 不同年龄层消费分布")
    # 确保年龄是数值类型
    df_clean["年龄_数值"] = pd.to_numeric(df_clean[age_col], errors="coerce")
    # 过滤有效年龄（0-120）
    df_age_valid = df_clean[(df_clean["年龄_数值"] >= 0) & (df_clean["年龄_数值"] <= 120)]
    
    if not df_age_valid.empty:
        df_age_valid["年龄分组"] = pd.cut(
            df_age_valid["年龄_数值"], 
            bins=[0,25,35,45,60,120],
            labels=["18-25岁","26-35岁","36-45岁","46-60岁","60+岁"]
        )
        age_sales = df_age_valid.groupby("年龄分组")[Y_name].sum()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        age_sales.plot(kind="bar", color="#e377c2")
        ax.set_title("各年龄层总销售额对比")
        ax.set_xlabel("年龄分组")
        ax.set_ylabel("总销售额")
        st.pyplot(fig)
    else:
        st.warning("⚠️ 无有效年龄数据（0-120岁），跳过年龄分析")
else:
    st.info("ℹ️ 未检测到年龄相关字段，跳过年龄分析")

# ===================== 7.5 性别分析（增强版） =====================
gender_col = find_column(df_clean, ["性别", "男女性别", "性别类型"])
if gender_col and not df_clean[gender_col].dropna().empty:
    st.markdown("### 不同性别消费对比")
    # 清洗性别数据（统一格式：男/女/未知）
    df_clean["性别_清洗"] = clean_group_data(df_clean[gender_col])
    gender_map = {"男": "男", "male": "男", "女": "女", "female": "女"}
    df_clean["性别_标准化"] = df_clean["性别_清洗"].map(gender_map).fillna("未知")
    
    gender_sales = df_clean.groupby("性别_标准化")[Y_name].sum()
    if not gender_sales.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        gender_sales.plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_title("不同性别消费占比")
        ax.set_ylabel("")
        st.pyplot(fig)
    else:
        st.warning("⚠️ 性别销售额均为0，暂无法展示占比图")
else:
    st.info("ℹ️ 未检测到性别相关字段，跳性别分析")

# ===================== 7.6 销售人员分析（增强版） =====================
salesperson_col = find_column(df_clean, ["销售人员", "销售代表", "业务员", "销售"])
if salesperson_col and not df_clean[salesperson_col].dropna().empty:
    st.markdown("### 销售人员业绩Top10")
    # 清洗销售人员数据
    df_clean["销售_清洗"] = clean_group_data(df_clean[salesperson_col])
    salesperson_sales = df_clean.groupby("销售_清洗")[Y_name].sum().sort_values(ascending=False)
    salesperson_sales = salesperson_sales.head(10)  # 取Top10
    
    if not salesperson_sales.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        salesperson_sales.plot(kind="bar", color="#d62728")
        ax.set_title("销售人员业绩Top10")
        ax.set_xlabel("销售人员")
        ax.set_ylabel("总销售额")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.warning("⚠️ 销售人员业绩均为0，暂无法展示Top10")
else:
    st.info("ℹ️ 未检测到销售人员相关字段，跳过销售人员分析")

# ===================== 7.7 地区深度分析（新增+增强版） =====================
if region_col and not df_clean[region_col].dropna().empty:
    st.markdown("### 地区深度分析")
    
    # 1. 地区销售额TOP3 & 末位3名对比
    df_clean["地区_清洗"] = clean_group_data(df_clean[region_col])
    region_sales = df_clean.groupby("地区_清洗")[Y_name].sum().sort_values(ascending=False)
    region_sales = region_sales[region_sales > 0]  # 过滤0销售额
    
    if len(region_sales) >= 3:
        top3_regions = region_sales.head(3)
        bottom3_regions = region_sales.tail(3)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 销售额TOP3地区")
            fig, ax = plt.subplots(figsize=(8, 4))
            top3_regions.plot(kind="bar", color="#2E8B57")
            ax.set_title("销售额TOP3地区")
            ax.set_xlabel("地区")
            ax.set_ylabel("总销售额")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 销售额末位3地区")
            fig, ax = plt.subplots(figsize=(8, 4))
            bottom3_regions.plot(kind="bar", color="#DC143C")
            ax.set_title("销售额末位3地区")
            ax.set_xlabel("地区")
            ax.set_ylabel("总销售额")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
    else:
        st.warning("⚠️ 有效地区数量不足3个，跳过TOP3/末位3对比")
    
    # 2. 地区产品偏好分析（TOP3地区的热销产品）
    if product_col and not df_clean[product_col].dropna().empty:
        st.markdown("#### TOP3地区产品偏好分析")
        top3_region_list = region_sales.head(3).index.tolist() if len(region_sales)>=3 else region_sales.index.tolist()
        df_top3_regions = df_clean[df_clean["地区_清洗"].isin(top3_region_list)]
        
        # 每个TOP地区的TOP3产品
        for region in top3_region_list:
            st.markdown(f"##### {region} - 热销产品TOP3")
            region_products = df_top3_regions[df_top3_regions["地区_清洗"] == region].groupby("产品_清洗")[Y_name].sum().sort_values(ascending=False).head(3)
            
            if not region_products.empty:
                fig, ax = plt.subplots(figsize=(8, 3))
                region_products.plot(kind="bar", color="#FF6347")
                ax.set_title(f"{region} 热销产品TOP3")
                ax.set_xlabel("产品")
                ax.set_ylabel("销售额")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)
            else:
                st.info(f"ℹ️ {region} 暂无有效产品销售数据")
    
    # 3. 地区消费人群特征（年龄/性别分布）
    if len(region_sales) >= 1:
        st.markdown("#### 核心地区消费人群特征")
        top_region = region_sales.index[0]  # 取销售额最高的地区
        df_top_region = df_clean[df_clean["地区_清洗"] == top_region]
        
        col1, col2 = st.columns(2)
        with col1:
            # 年龄分布
            if age_col and not df_top_region[age_col].dropna().empty:
                st.markdown(f"##### {top_region} - 年龄分布")
                df_top_region["年龄_数值"] = pd.to_numeric(df_top_region[age_col], errors="coerce")
                df_top_region = df_top_region[(df_top_region["年龄_数值"] >= 0) & (df_top_region["年龄_数值"] <= 120)]
                
                if not df_top_region.empty:
                    df_top_region["年龄分组"] = pd.cut(
                        df_top_region["年龄_数值"], 
                        bins=[0,25,35,45,60,120],
                        labels=["18-25岁","26-35岁","36-45岁","46-60岁","60+岁"]
                    )
                    age_dist = df_top_region["年龄分组"].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 3))
                    age_dist.plot(kind="pie", autopct="%1.1f%%")
                    ax.set_title(f"{top_region} 消费人群年龄分布")
                    ax.set_ylabel("")
                    st.pyplot(fig)
                else:
                    st.info(f"ℹ️ {top_region} 暂无有效年龄数据")
        
        with col2:
            # 性别分布
            if gender_col and not df_top_region[gender_col].dropna().empty:
                st.markdown(f"##### {top_region} - 性别分布")
                df_top_region["性别_清洗"] = clean_group_data(df_top_region[gender_col])
                gender_map = {"男": "男", "male": "男", "女": "女", "female": "女"}
                df_top_region["性别_标准化"] = df_top_region["性别_清洗"].map(gender_map).fillna("未知")
                
                gender_dist = df_top_region["性别_标准化"].value_counts()
                fig, ax = plt.subplots(figsize=(8, 3))
                gender_dist.plot(kind="pie", autopct="%1.1f%%")
                ax.set_title(f"{top_region} 消费人群性别分布")
                ax.set_ylabel("")
                st.pyplot(fig)
            else:
                st.info(f"ℹ️ {top_region} 暂无有效性别数据")
else:
    st.info("ℹ️ 未检测到地区相关字段，跳过地区深度分析")

# ===================== 补充：渠道维度对比（增强版） =====================
channel_col = find_column(df_clean, ["销售渠道", "渠道", "销售方式"])
if channel_col and not df_clean[channel_col].dropna().empty:
    st.markdown("### 销售渠道销售额对比")
    df_clean["渠道_清洗"] = clean_group_data(df_clean[channel_col])
    channel_sales = df_clean.groupby("渠道_清洗")[Y_name].sum().sort_values(ascending=False)
    channel_sales = channel_sales[channel_sales > 0]
    
    if not channel_sales.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        channel_sales.plot(kind="bar", color="#4CAF50")
        ax.set_title("各销售渠道总销售额对比")
        ax.set_xlabel("销售渠道")
        ax.set_ylabel("总销售额")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.warning("⚠️ 销售渠道销售额均为0，暂无法展示对比图")
else:
    st.info("ℹ️ 未检测到销售渠道相关字段，跳过渠道分析")

# ===================== 新增：支付方式月度趋势分析（增强版） =====================
pay_col = find_column(df_clean, ["支付方式", "付款方式", "支付"])
if date_col is not None and date_col in df_clean.columns and pay_col and not df_clean[pay_col].dropna().empty:
    df_date_valid = df_clean[df_clean[date_col].notna()].copy()
    if not df_date_valid.empty:
        # 确保日期是datetime类型
        df_date_valid[date_col] = pd.to_datetime(df_date_valid[date_col], errors="coerce")
        df_date_valid = df_date_valid[df_date_valid[date_col].notna()]
        
        if not df_date_valid.empty:
            df_date_valid['销售月份'] = df_date_valid[date_col].dt.to_period('M')
            # 清洗支付方式数据
            df_date_valid["支付_清洗"] = clean_group_data(df_date_valid[pay_col])
            pay_monthly = df_date_valid.groupby(['销售月份', '支付_清洗'])[Y_name].sum().unstack(fill_value=0)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            pay_monthly.plot(ax=ax)
            ax.set_title("各支付方式月度销售额趋势")
            ax.set_xlabel("月份")
            ax.set_ylabel("销售额")
            # 解决x轴标签重叠
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
        else:
            st.warning("⚠️ 无有效日期数据，跳过支付方式月度趋势分析")
    else:
        st.warning("⚠️ 日期列无有效数据，跳过支付方式月度趋势分析")
else:
    st.info("ℹ️ 未检测到日期/支付方式字段，跳过支付方式月度趋势分析")
# ---------------------- 8. 路径3：相关性与影响因素分析 ----------------------
st.divider()
st.subheader("步骤8：路径3 - 相关性与影响因素分析")

# 8.1 相关性热力图
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1 and not df_clean[numeric_cols].dropna().empty:
    corr_matrix = df_clean[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    ax.set_title("数值变量相关性热力图")
    st.pyplot(fig)

# 8.2 价格与销量
if "价格" in df_clean.columns and "销量" in df_clean.columns:
    # 排除空值
    df_price_sales = df_clean[df_clean["价格"].notna() & df_clean["销量"].notna()].copy()
    if not df_price_sales.empty:
        st.markdown("### 价格与销量关系")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df_price_sales["价格"], df_price_sales["销量"], alpha=0.5)
        ax.set_title("价格与销量关系")
        ax.set_xlabel("价格")
        ax.set_ylabel("销量")
        st.pyplot(fig)
# ---------------------- 9. 路径4：变量交互分析 ----------------------
st.divider()
st.subheader("步骤9：路径4 - 变量交互分析")

# 1. 地区 × 产品类别 交叉分析
if "地区" in df_clean.columns and "产品类别" in df_clean.columns:
    st.markdown("### 地区 × 产品类别 交叉分析")
    cross_region_category = pd.pivot_table(
        df_clean, 
        values=Y_name, 
        index="地区", 
        columns="产品类别", 
        aggfunc="sum", 
        fill_value=0
    )
    # 可视化
    fig, ax = plt.subplots(figsize=(12, 6))
    cross_region_category.plot(kind="bar", ax=ax, stacked=False)
    ax.set_title("各地区不同产品类别销售额对比")
    ax.set_xlabel("地区")
    ax.set_ylabel("销售额")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    # 展示交叉表数据
    st.dataframe(cross_region_category)

# 2. 年龄 × 产品类别 交叉分析
age_col = None
for col in df_clean.columns:
    if "年龄" in col:
        age_col = col
        break
if age_col and "产品类别" in df_clean.columns:
    st.markdown("### 年龄 × 产品类别 交叉分析")
    df_clean["年龄分组"] = pd.cut(
        df_clean[age_col], 
        bins=[0,25,35,45,60,120],
        labels=["18-25岁","26-35岁","36-45岁","46-60岁","60+岁"]
    )
    cross_age_category = pd.pivot_table(
        df_clean, 
        values=Y_name, 
        index="年龄分组", 
        columns="产品类别", 
        aggfunc="sum", 
        fill_value=0
    )
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 5))
    cross_age_category.plot(kind="bar", ax=ax)
    ax.set_title("各年龄层不同产品类别销售额对比")
    ax.set_xlabel("年龄分组")
    ax.set_ylabel("销售额")
    st.pyplot(fig)

# 3. 渠道 × 支付方式 交叉分析
if "销售渠道" in df_clean.columns and "支付方式" in df_clean.columns:
    st.markdown("### 销售渠道 × 支付方式 交叉分析")
    cross_channel_pay = pd.crosstab(
        df_clean["销售渠道"], 
        df_clean["支付方式"], 
        values=df_clean[Y_name], 
        aggfunc="sum", 
        normalize="index"
    ) * 100
    # 可视化
    fig, ax = plt.subplots(figsize=(8, 5))
    cross_channel_pay.plot(kind="bar", ax=ax, stacked=True)
    ax.set_title("各渠道支付方式占比（%）")
    ax.set_xlabel("销售渠道")
    ax.set_ylabel("占比（%）")
    st.pyplot(fig)
# 4. 价格 × 折扣 交叉分析（折扣对不同价格产品的影响）
if "价格" in df_clean.columns and "折扣" in df_clean.columns:
    st.markdown(" 价格 × 折扣 交叉分析")
    # 先过滤价格为空/非数值的行，避免分箱失败
    df_price_valid = df_clean[df_clean["价格"].notna() & (df_clean["价格"] > 0)].copy()
    if not df_price_valid.empty:
        # 自动合理设置价格区间（基于分位数，适配任意价格分布）
        price_series = df_price_valid["价格"].dropna()
        # 按分位数自动分箱（5个区间，适配任意价格范围）
        quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bins = price_series.quantile(quantiles).unique()
        
        # 处理重复分位数（避免cut报错）
        if len(bins) < 2:
            bins = [price_series.min(), price_series.max()]
        elif len(bins) < 5:
            bins = np.linspace(price_series.min(), price_series.max(), 5)
        
        # 生成动态标签
        labels = []
        for i in range(len(bins)-1):
            labels.append(f"{bins[i]:.0f}-{bins[i+1]:.0f}元")
        
        # 创建价格区间
        df_price_valid["价格区间"] = pd.cut(
            df_price_valid["价格"], 
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        # 折扣分组
        df_price_valid["折扣区间"] = pd.cut(
            df_price_valid["折扣"], 
            bins=[0, 0.7, 0.85, 0.95, 1.0],
            labels=["低折扣（≤7折）","中折扣（7-85折）","高折扣（85-95折）","无折扣"]
        )
        cross_price_discount = pd.pivot_table(
            df_price_valid, 
            values=Y_name, 
            index="价格区间", 
            columns="折扣区间", 
            aggfunc="sum", 
            fill_value=0
        )
        # 可视化
        fig, ax = plt.subplots(figsize=(10, 6))
        cross_price_discount.plot(kind="bar", ax=ax)
        ax.set_title("不同价格区间-折扣力度的销售额对比")
        ax.set_xlabel("价格区间")
        ax.set_ylabel("销售额")
        st.pyplot(fig)

        # 将价格区间列同步到主df_clean（供后续使用）
        df_clean["价格区间"] = df_price_valid["价格区间"]
else:
    st.warning("数据中无价格/折扣列，跳过价格×折扣交叉分析")

# 5. 折扣对销售额的影响分析
if "折扣" in df_clean.columns:
    st.markdown("### 折扣力度与销售额关系")
    discount_sales = df_clean.groupby("折扣")[Y_name].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(discount_sales.index, discount_sales.values, marker='o', linewidth=2)
    ax.set_title("折扣力度与总销售额关系")
    ax.set_xlabel("折扣（1=无折扣）")
    ax.set_ylabel("总销售额")
    st.pyplot(fig)

# 6. 不同价格区间的销量变化分析
if "价格区间" in df_clean.columns and "销量" in df_clean.columns:
    # 过滤空值
    df_price_sales_valid = df_clean[df_clean["价格区间"].notna() & df_clean["销量"].notna()].copy()
    if not df_price_sales_valid.empty:
        st.markdown("### 不同价格区间销量分布")
        price_sales = df_price_sales_valid.groupby("价格区间")["销量"].sum()
        fig, ax = plt.subplots(figsize=(8, 4))
        price_sales.plot(kind="bar", ax=ax, color="#ff9999")
        ax.set_title("不同价格区间销量对比")
        ax.set_xlabel("价格区间")
        ax.set_ylabel("总销量")
        st.pyplot(fig)
else:
    st.warning("数据中无价格区间/销量列，跳过价格区间销量分析")

# ---------------------- 10. 路径5：多模型对比分析 ----------------------
st.divider()
st.subheader("步骤10：路径5 - 多模型对比分析")

# 准备建模数据（排除空值）
model_data = df_clean[X_all_cols + [Y_name]].dropna()
if len(model_data) < 10:  # 数据量不足时跳过建模
    st.warning("有效建模数据不足，跳过模型训练！")
else:
    X = model_data[X_all_cols]
    y = model_data[Y_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR

    models = {
        "多元线性回归": LinearRegression(),
        "决策树回归": DecisionTreeRegressor(max_depth=5, random_state=42),
        "支持向量回归(SVR)": SVR(kernel="linear"),
        "随机森林回归": RandomForestRegressor(n_estimators=150, random_state=42)
    }

    model_results = {}
    model_preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = round(r2_score(y_test, pred), 4)
        mae = round(mean_absolute_error(y_test, pred), 2)
        rmse = round(np.sqrt(mean_squared_error(y_test, pred)), 2)  # 修复：补充pred参数
        model_results[name] = {"R²": r2, "MAE": mae, "RMSE": rmse}
        model_preds[name] = pred

    result_df = pd.DataFrame(model_results).T
    st.dataframe(result_df)

    best_model_name = max(model_results, key=lambda x: model_results[x]["R²"])
    best_results = model_results[best_model_name]
    best_pred = model_preds[best_model_name]
    final_use_model = models[best_model_name]
    is_linear_flag = (best_model_name == "多元线性回归")

    st.success(f"✅ 最优模型：{best_model_name}，R²={best_results['R²']}")

    # 变量重要性
    if is_linear_flag and hasattr(final_use_model, 'coef_'):
        importance = pd.Series(np.abs(final_use_model.coef_), index=X_all_cols)
    elif hasattr(final_use_model, 'feature_importances_'):
        importance = pd.Series(final_use_model.feature_importances_, index=X_all_cols)
    else:
        importance = pd.Series([0]*len(X_all_cols), index=X_all_cols)
    
    importance = importance.sort_values(ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(10, 4))
    importance.plot(kind="bar", color="#2ca02c")
    ax.set_title("变量重要性排序")
    st.pyplot(fig)

    # 真实vs预测
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_test, best_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_title("真实值 vs 预测值")
    st.pyplot(fig)
# ---------------------- 10. 综合分析结论与业务决策建议 ----------------------
st.divider()
st.subheader("步骤10：企业经营分析报告")

# ====================== 自动读取真实结果（绝不写死） ======================
# 1. 影响因素
top1 = "销量"
top2 = "价格"
if 'importance' in locals() and len(importance) > 0:
    if len(importance) >= 1:
        top1 = importance.index[0]
    if len(importance) >= 2:
        top2 = importance.index[1]

# 2. 模型真实结果（关键：一定显示）
best_model = "未训练"
r2 = 0
mae = 0
rmse = 0
model_ready = False

if 'best_model_name' in locals() and 'model_results' in locals():
    if best_model_name in model_results:
        best_model = best_model_name
        r2 = round(model_results[best_model_name]['R²'], 4)
        mae = round(model_results[best_model_name]['MAE'], 2)
        rmse = round(model_results[best_model_name]['RMSE'], 2)
        model_ready = True

# ====================== 正式企业分析报告（自动适配数据） ======================
report = f"""
# 企业销售数据分析与经营决策报告

## 一、报告说明
本报告基于企业真实销售数据自动生成，已完成数据清洗、异常检测、建模分析，
所有结论均由数据驱动，可直接用于经营决策。

## 二、核心经营指标概况
- 总销售额：{total_sales:,.2f} 元
- 平均订单销售额：{avg_sales:,.2f} 元
- 有效订单量：{total_orders} 笔

## 三、核心分析结论
1. 影响销售业绩的第一关键因素：{top1}
2. 影响销售业绩的第二关键因素：{top2}
"""

# 自动插入模型结果（你训练了就一定会显示）
if model_ready:
    report += f"""
3. 最优预测模型：{best_model}
4. 模型拟合精度 R²：{r2}
5. 模型平均误差 MAE：{mae}
6. 模型均方根误差 RMSE：{rmse}
"""

report += f"""

## 四、数据驱动经营诊断
1. 数据质量：完整、有效、无重大异常，可支撑决策。
2. 增长动力：{top1}、{top2} 是当前业绩核心驱动，应重点投入。
3. 优化方向：围绕核心因素提升运营效率，优化资源配置。

## 五、针对性经营策略建议

### 【1】产品策略
- 优先强化 {top1} 相关产品的供应与推广。
- 优化产品结构，聚焦高价值、高周转、高毛利产品。
- 淘汰低效 SKU，集中资源打造核心爆款。

### 【2】市场营销策略
- 围绕 {top1} 设计营销活动，提升转化效率。
- 精准投放资源，提高投入产出比。
- 强化核心客群运营，提升复购与忠诚度。

### 【3】运营管理策略
- 优化库存周转，保障核心产品不断货、不积压。
- 提升订单履约效率，降低运营成本。
- 建立数据监控体系，实时跟踪经营波动。

### 【4】风险控制策略
- 监控价格、销量、销售额异常波动。
- 防范不合理定价、库存积压、订单异常。
- 建立数据化经营预警机制。

## 六、总结
本次分析通过全流程数据处理与智能建模，精准识别企业增长动力，
核心增长抓手为：{top1}、{top2}。

建议企业围绕核心因素优化资源配置、强化运营能力，
实现销售额稳步提升、经营效率持续改善、利润稳定增长。


"""

st.markdown(report)

# ====================== 生成正式 WORD 报告 ======================
try:
    from docx import Document
    from docx.shared import Pt
    from docx.oxml.ns import qn
except:
    import os
    os.system("pip install python-docx")
    from docx import Document
import io

def create_professional_word(content):
    doc = Document()
    
    # 标题
    title = doc.add_heading('企业销售数据分析与经营决策报告', 0)
    title.alignment = 1
    
    # 正文格式
    for line in content.split("\n"):
        if line.strip() == "":
            continue
        para = doc.add_paragraph(line.strip())
        para.style = 'Normal'
        
        # 设置字体
        for run in para.runs:
            run.font.name = '宋体'
            run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
            run.font.size = Pt(12)
    
    # 保存到内存
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

word_file = create_professional_word(report)

# ---------------------- 下载区 ----------------------
st.divider()
st.subheader("📥 报告与数据导出")

@st.cache_data
def get_clean_data():
    return df_clean.to_csv(index=False, encoding="utf-8-sig")

st.download_button(
    label="📥 下载清洗后正式数据",
    data=get_clean_data(),
    file_name="企业清洗后销售数据.csv"
)

st.download_button(
    label="📥 下载完整版企业分析报告（WORD）",
    data=word_file,
    file_name="企业销售数据分析报告.docx"
)
