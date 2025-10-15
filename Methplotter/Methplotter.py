import subprocess
import sys

packages = ['pandas', 'numpy', 'seaborn', 'matplotlib','pyqt5']
for package in packages:
    try:
        # 尝试导入包，检查是否已安装
        __import__(package)
        print(f"{package} 已经安装")
    except ImportError:
        print(f"{package} 未安装，正在安装...")
        # 使用 pip 安装包
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} 安装成功")

import pandas
import numpy
import seaborn
import matplotlib
import os

matplotlib.use('Qt5Agg')  # 或 'TkAgg'
import matplotlib.pyplot as plt

# 设置Pandas的显示选项
pandas.set_option('display.max_columns', None)  # 显示所有列
pandas.set_option('display.width', 2000)  # 设置每行的宽度


#将原始CX_report文件根据chr或者context或者chrAndcontext分类成小文件
def ClassifyByBase(CX_report_df,base):
    # 参数：
    # CX_report_df：原始CX_report.df，pandas df格式
    # save_path: 输出文件保存地址
    # base：选择根据chr还是context来进行分类，chr-->[Chr1,Chr2,Chr3...], context-->[CHH,CHG,CG]

    # 功能代码
    # 检查base参数是否有效
    if base not in ["chr", "context",'chrAndcontext']:
        raise ValueError("base 参数必须是 'chr' 或 'context'或‘chrAndcontext")

    #  初始化一个空列表来存储分类后的DataFrame
    classified_dfs = []
    # 根据base参数进行分类
    if base == "chr":
        # 按染色体分类
        for chr_value in CX_report_df['chr'].unique():
            chr_df = CX_report_df[CX_report_df['chr'] == chr_value]
            # 保存每个染色体的DataFrame为CSV文件
            # chr_df.to_csv(os.path.join(save_path, f'chr_{chr_value}.csv'), index=False)
            classified_dfs.append(chr_df)
    elif base == "context":
        # 按上下文分类
        for context_value in CX_report_df['context'].unique():
            context_df = CX_report_df[CX_report_df['context'] == context_value]
            # 保存每个上下文的DataFrame为CSV文件
            # context_df.to_csv(os.path.join(save_path, f'context_{context_value}.csv'), index=False)
            classified_dfs.append(context_df)
    elif base == "chrAndcontext":
        # 同时按chr和context分类
        for chr_value in CX_report_df['chr'].unique():
            for context_value in CX_report_df['context'].unique():
                # 筛选出当前chr和context的DataFrame
                filtered_df = CX_report_df[
                    (CX_report_df['chr'] == chr_value) & (CX_report_df['context'] == context_value)]
                classified_dfs.append(filtered_df)
    print('分类并保存文件完成')
    # 返回一个列表，列表里装了df
    return classified_dfs

# 将df文件进行数据处理，根据分组，每组ratio取平均值，返回平均值df
def ClassifyBySize(CX_report_specific_df,min_Pos,max_Pos ,window_size):
    # 找到 pos 的最小值和最大值
    min_pos = min_Pos
    max_pos = max_Pos
    # print(min_Pos)
    # print(max_Pos)
    df = CX_report_specific_df.copy()
    # 获取context类型
    context= df.iloc[0]['context']
    # 生成完整的 pos 序列
    all_pos = list(range(min_pos, max_pos + 1))
    # print(all_pos)
    # # 将 pos 列转换为类别，以确保所有 pos 都被包含
    # df['pos'] = pandas.Categorical(df['pos'], categories=all_pos, ordered=True)
    # # print(df)
    # # 使用 reindex 方法补充缺失的 pos
    # df = df.set_index('pos').reindex(all_pos).reset_index()

    # 创建一个包含所有 pos 的模板 DataFrame
    template = pandas.DataFrame({'pos': all_pos})
    # 将 df 与模板合并，保留所有 pos
    df = pandas.merge(template, df, on='pos', how='left')

    # 将缺失的 ratio 填充为 0
    df['ratio'] = df['ratio'].fillna(0)

    # 将非 'CHH' 的 ratio 设置为 0（使用 .loc 显式赋值）
    df.loc[df['context'] != context, 'ratio'] = 0  # 替换这里
    # print(df)
    # 每 window_size 行计算一次平均值
    df['group'] = df.index // window_size  # 创建分组索引

    # 计算每组的 ratio 平均值
    result_df = df.groupby('group')['ratio'].mean().reset_index(drop=True)

    # series转换成df
    result_df = result_df.to_frame(name='ratio')  # 给结果列命名

    # 添加一列 'Index'，值为原索引的 window_size 倍
    result_df['Index'] = result_df.index * window_size

    # 将 'Index' 列移动到前面
    result_df = result_df[['Index', 'ratio']]

    return result_df

# 将df文件进行数据处理，根据分组，每组ratio取平均值，返回平均值df(想要得到多少个分组)
def ClassifyToGivenSize(CX_report_specific_df, given_size):
    num_rows = len(CX_report_specific_df)
    # 创建一个包含num_rows个元素的数组，用作组标识符
    group_ids = numpy.arange(num_rows)
    # 计算每组的大小，使用-1来自动计算剩余的元素    group_size是 商，remainder是余数
    group_size, remainder = divmod(num_rows, given_size)
    # 创建一个包含组标识符的数组
    groups = [group_size] * given_size
    # 将剩余的元素加到最后一组
    groups[-1] += remainder
    # 将数组分割成对应的组
    split_ids = numpy.split(group_ids, numpy.cumsum(groups)[:-1])
    # 对每个组应用平均值计算
    averages = []
    for i, group in enumerate(split_ids):
        group_average = CX_report_specific_df.iloc[group]['ratio'].mean()
        averages.append((i, group_average))
    # 将结果转换为DataFrame
    averages_df = pandas.DataFrame(averages, columns=['Group', 'Value'])
    # 添加一个新列，该列的值就是原来的索引
    averages_df['Index'] = averages_df.index * group_size
    return averages_df

# # 检索 attributes 列中包含 gene_string 且 type 属性为 'gene' 的行
# def SearchByName(Gff_file,gene_string):
#     #参数：
#     # Gff_file: gff_file ，df格式
#     # gene_string：基因的名字，string
#
#     # case=False：表示搜索时不区分大小写。
#     # na=False：表示如果attributes列中的值是NaN，则返回False，不进行搜索。
#     filtered_rows = Gff_file[(Gff_file['attributes'].str.contains(gene_string, case=False, na=False)) & (Gff_file['type'] == 'gene')]
#
#     return filtered_rows

# # 获得gff_specific_df的seqid，start，end，strand值
# def get_seqid_start_end(filtered_rows):
#     # 提取 seqid, start, end ，strand属性的值
#     result = filtered_rows[['seqid', 'start', 'end','strand']].copy()
#     # 重置索引，因为原始的filtered_rows可能保留了gff_file的索引
#     result.reset_index(drop=True, inplace=True)
#
#     return result

# 根据result的四个值seqid，start，end，strand值 ,对应CX_file_df的数据
def filter_CX_file(CX_file_df, result):
    # 确保 result 是一个 DataFrame
    if not isinstance(result, pandas.DataFrame):
        raise ValueError("result 必须是 DataFrame 类型")

    # 定义筛选条件
    conditions = [
        CX_file_df['chr'] == result['seqid'].iloc[0],  # 假设 result 中只有一个 seqid 值
        # CX_file_df['strand'] == result['strand'].iloc[0],  # 假设 result 中只有一个 strand 值
        (CX_file_df['pos'] >= result['start'].iloc[0]) & (CX_file_df['pos'] <= result['end'].iloc[0])
        # 假设 result 中只有一个 start 和 end 值
    ]
    # 应用筛选条件
    # filtered_CX = CX_file_df[conditions[0] & conditions[1] & conditions[2]]
    filtered_CX = CX_file_df[conditions[0] & conditions[1]]
    return filtered_CX

# 函数 筛选出基因片段上下1kb的片段
def filter_CX_file_UpAndDown(CX_file_df, result,length):
    # 参数：
    # CX_file_df：CX_file文件
    # result： 经过get_seqid_start_end得到的df
    # length：想要筛选的上有和下游长度,int 正数

    # 确保 result 是一个 DataFrame
    if not isinstance(result, pandas.DataFrame):
        raise ValueError("result 必须是 DataFrame 类型")
    # 定义筛选条件
    seqid = result['seqid'].iloc[0]  # 获取 seqid 值
    # strand = result['strand'].iloc[0]  # 获取 strand 值
    start = result['start'].iloc[0]  # 获取 start 值
    end = result['end'].iloc[0]  # 获取 end 值
    # 计算前1kb和后1kb的范围
    upstream = start - int(length)  # 上游1kb
    downstream = end + int(length)  # 下游1kb

    # 生成完整的 pos 序列
    all_pos = list(range(upstream, downstream + 1))
    # print(all_pos)
    # 创建一个包含所有 pos 的模板 DataFrame
    template = pandas.DataFrame({'pos': all_pos})
    # 将 df 与模板合并，保留所有 pos
    CX_file_df = pandas.merge(template, CX_file_df, on='pos', how='left')

    # 将缺失的 ratio 填充为 0
    CX_file_df['ratio'] = CX_file_df['ratio'].fillna(0)

    # 定义筛选条件
    conditions = [
        CX_file_df['chr'] == seqid,  # 染色体位置匹配
        # CX_file_df['strand'] == strand,  # 链的方向匹配
        (CX_file_df['pos'] >= upstream) & (CX_file_df['pos'] <= downstream)  # 位置在基因片段前1kb至后1kb范围内
    ]
    # 应用筛选条件
    # filtered_CX = CX_file_df[conditions[0] & conditions[1] & conditions[2]]
    filtered_CX = CX_file_df[conditions[0] & conditions[1] ]

    return filtered_CX

# ---------------------------------------------------------------------------------------

# 根据gff文件生成相应的可以进行绘图（折线图）的文件
def GenerateFileToDrawLineplot(CX_file):

    # 参数
    # CX_file：cx文件，df格式
    # 返回一个df，里面装了很多个df
    df =ClassifyByBase(CX_file,'chr')
    length = len(df)
    print('一共有 '+str(length)+' 条染色体')
    return df

# 根据gff文件生成相应的可以进行绘图（折线图）的文件
def GenerateFileToDrawLineExtraplot(CX_file,gff_file,UpAndDown):
    # 参数
    # CX_file：cx文件，df格式
    # gff_file：gff文件,df格式

    # 创建一个空的DataFrame
    seqid = gff_file.iloc[0].seqid
    start = gff_file.start.min()
    end = gff_file.end.max()
    tem2 = pandas.DataFrame(columns=['seqid', 'start', 'end', 'strand'])
    # 添加一行数据
    tem2.loc[0] = [seqid , start , end, '+']

    df = filter_CX_file_UpAndDown(CX_file, tem2,UpAndDown)

    return  df

# 根据gff文件生成相应的可以进行绘图（聚类绘图）的文件
def GenerateFileToDrawClusterplot(CX_file,gene_list,given_size):
    # 参数
    # CX_file：cx文件，df格式
    # gff_file：gff文件,df格式
    # list：一个装有字符串的列表，这些字符串是你想要筛选的片段gff [fragemnt1 ,fragment2, fragment3]
    # given_size: 将片段分成多少组进行取平均

    # 功能代码
    # 创建一个空的DataFrame
    df = pandas.DataFrame()
    name_list = []
    for i in range(0,len(gene_list)):
        # 以下两行是功能修改前的代码，及时删除
        # tem1 = SearchByName(gff_file, gene_list[i])
        # tem2 = get_seqid_start_end(tem1)

        seqid = gene_list[i].iloc[0].seqid
        name = gene_list[i].iloc[0].phrase_name
        start = gene_list[i].start.min()
        end = gene_list[i].end.max()
        name_list.append(name)
        tem2 = pandas.DataFrame(columns=['seqid', 'start', 'end', 'strand'])
        # 添加一行数据
        tem2.loc[0] = [seqid, start, end, '+']

        tem3 = filter_CX_file(CX_file, tem2)
        tem4 = ClassifyToGivenSize(tem3 , given_size)
        tem5 = pandas.DataFrame([tem4['Value']])
        # 合并DataFrame
        df = pandas.concat([df, tem5], ignore_index=True)

    # 修改列名
    df = df.T
    df.columns = name_list
    # 将所有空值替换为 0
    df = df.fillna(0)
    return df #返回一个倒置df

# 根据gff文件生成相应的可以进行绘图（小提琴图）的文件
def GenerateFileToDrawViolinplot(CX_file,gff_file_list,context):
    # 参数
    # CX_file：cx文件，df格式
    # gff_file_list：list格式，里面装了多个gff文件
    # context: CG CHG CHH

    # 创建一个空的DataFrame
    df = pandas.DataFrame()
    for i in range(0,len(gff_file_list)):
        # 创建一个空的DataFrame
        seqid = gff_file_list[i].iloc[0].seqid
        start = gff_file_list[i].start.min()
        end = gff_file_list[i].end.max()
        tem2 = pandas.DataFrame(columns=['seqid', 'start', 'end', 'strand'])
        # 添加一行数据
        tem2.loc[0] = [seqid, start, end, '+']
        # 进一步操作
        tem3 = filter_CX_file(CX_file, tem2)
        tem4=tem3[tem3['context']==context]
        # 提取ratio列的值
        ratio_values = tem4['ratio'].tolist()
        # 创建tem DataFrame
        tem_df = pandas.DataFrame({
            'gene': [gff_file_list[i].iloc[0].phrase_name] * len(ratio_values),
            'ratio': ratio_values
        })
        df = pandas.concat([df, tem_df], ignore_index=True)

    return df

# 根据gff文件生成相应的可以进行绘图（柱状图）的文件
def GenerateFileToDrawBarplot(CX_file,gff_file_list,context):
    # 参数
    # CX_file：cx文件，df格式
    # gff_file_list：list格式，里面装了多个gff文件
    # context: CG CHG CHH

    # 创建一个空的DataFrame
    df = pandas.DataFrame()
    for i in range(0,len(gff_file_list)):
        # print('i =' + str(i))
        # 创建一个空的DataFrame
        seqid = gff_file_list[i].iloc[0].seqid
        start = gff_file_list[i].start.min()
        end = gff_file_list[i].end.max()
        tem2 = pandas.DataFrame(columns=['seqid', 'start', 'end', 'strand'])
        # 添加一行数据
        tem2.loc[0] = [seqid, start, end, '+']
        # print(tem2)
        # 进一步操作
        tem3 = filter_CX_file(CX_file, tem2)
        # print(tem3)
        tem4=tem3[tem3['context']==context]
        # print(tem4)
        # 提取ratio列的值
        ratio_values = tem4['ratio'].tolist()
        # print(ratio_values)
        # 创建tem DataFrame
        tem_df = pandas.DataFrame({
            'gene': [gff_file_list[i].iloc[0].phrase_name] * len(ratio_values),
            'ratio': ratio_values
        })
        # print(tem_df)
        df = pandas.concat([df, tem_df], ignore_index=True)
        # print(df)
    return df

# ----------------------------------------------------------------------------------------

# 将绘图功能包装成函数  绘制折线图
def Draw_line(CX_report_list, index,window_size,color):
    # 参数:
    # CX_report_df_list: 输入文件，列表里装了多个df
    # window_size: 窗口大小，整数格式
    # color: 绘制点的颜色,列表格式
    # index : 选择CX_report_df_list的第几行的参数，int
    # save_path: 输出文件保存地址，字符串格式

    # 功能代码
    CX_report_df_tem = CX_report_list[index]
    window_size_tem = window_size
    chromosome_name = CX_report_df_tem.iloc[0]['chr']
    # 将CX_report_df_tem列表中的每一个文件进行操作

    # 提取每个 context 对应的 DataFrame
    df_cg = CX_report_df_tem[CX_report_df_tem["context"] == "CG"]
    df_chh = CX_report_df_tem[CX_report_df_tem["context"] == "CHH"]
    df_chg = CX_report_df_tem[CX_report_df_tem["context"] == "CHG"]

    # 将这 3 个 DataFrame 存入一个列表
    df_list = [df_cg, df_chh, df_chg]
    min_pos = CX_report_df_tem.iloc[0]['pos']
    max_pos = CX_report_df_tem.iloc[-1]['pos']
    # 使用列表推导式对每个元素应用函数
    averages_df = [ClassifyBySize(x,min_pos,max_pos,window_size_tem) for x in df_list]

    # 绘制点图，使用新创建的'Index'列作为横坐标，'Value'列作为纵坐标
    # 创建一个图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))
    # 在坐标轴上绘制每个散点图
    for i in range(len(averages_df)):
        ax.plot(averages_df[i]['Index'], averages_df[i]['ratio'],
                color=color[i], label='CG' if i == 0 else ('CHG' if i == 1 else 'CHH'))
    # 设置图形的标题和坐标轴标签
    plt.title("".join(['chromosome ', chromosome_name]))
    plt.xlabel('Genomic Position')
    plt.ylabel('Methylation Level')
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()

    return 0

# 将绘图功能包装成函数  绘制折线图(附带基因前后段)
def Draw_lineExtra(CX_file, gene_string,UpAndDown,window_size,color):
    # 参数:
    # CX_report_df_list: 输入文件，df格式
    # window_size: 窗口大小，整数格式
    # color: 绘制点的颜色,列表格式
    # save_path: 输出文件保存地址，字符串格式

    # 功能代码
    gene_phrase = gene_string
    extent = UpAndDown
    CX_report_df_tem = CX_file
    window_size_tem = window_size
    # 提取每个 context 对应的 DataFrame
    df_cg = CX_report_df_tem[CX_report_df_tem["context"] == "CG"]
    df_chh = CX_report_df_tem[CX_report_df_tem["context"] == "CHH"]
    df_chg = CX_report_df_tem[CX_report_df_tem["context"] == "CHG"]

    # 将这 3 个 DataFrame 存入一个列表
    df_list = [df_cg, df_chh, df_chg]

    min_pos = CX_report_df_tem.iloc[0]['pos']
    max_pos = CX_report_df_tem.iloc[-1]['pos']
    # 使用列表推导式对每个元素应用函数
    averages_df = [ClassifyBySize(x, min_pos,max_pos,window_size_tem) for x in df_list]

    # 创建一个图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))
    # 在坐标轴上绘制每个散点图
    for i in range(len(averages_df)):
        ax.plot(averages_df[i]['Index'], averages_df[i]['ratio'],
                color=color[i], label='CG' if i == 0 else ('CHG' if i == 1 else 'CHH'))
    # 假设我们想在 区域填充背景颜色
    end = int(averages_df[0].iloc[-1]['Index']) - int(extent)
    ax.axvspan(int(extent), end, color='gray', alpha=0.3)
    # 设置图形的标题和坐标轴标签
    plt.title("".join([gene_phrase, ' methylation level over gene body']))
    plt.xlabel('gene body')
    plt.ylabel('Methylation Level')

    # 设置新的刻度位置和标签
    # 假设横坐标结尾的值为 max_index
    max_index = averages_df[0].iloc[-1]['Index']
    ax.set_xticks([0, max_index])
    ax.set_xticklabels([''.join(['-',str(extent/1000)]), ''.join([str(extent/1000),'kb'])])
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()

# 将绘图功能包装成函数 绘制聚类热图
def Draw_cluster(file, cmap_kind, row_cluster, col_cluster):
    # 绘制聚类热图
    g = seaborn.clustermap(file, cmap=cmap_kind, annot=True, fmt=".2f", figsize=(10, 10), row_cluster=row_cluster,
                       col_cluster=col_cluster,cbar_kws={#'label': 'ColorbarName', #color bar的名称
                           'orientation': 'horizontal',#color bar的方向设置，默认为'vertical'，可水平显示'horizontal'
                           # "ticks":numpy.arange(0,1,0.5),#color bar中刻度值范围和间隔
                           # "format":"%.1f",#格式化输出color bar中刻度值
                           # "pad":0.15,#color bar与热图之间距离，距离变大热图会被压缩
                                                   },
                           cbar_pos=(0.2,0.9,0.7,0.02)
                           )
    # 清除横纵坐标的刻度
    # g.ax_heatmap.set_xticks([])  # 清除横坐标的刻度
    g.ax_heatmap.set_yticks([])  # 清除纵坐标的刻度

# 将绘图功能包装成函数  绘制箱型图
def Draw_violin(file,context):
    # 使用seaborn绘制箱型图
    seaborn.violinplot(x='gene', y='ratio', data=file,
    palette='pastel',  # 使用柔和的配色方案
    linewidth=1,  # 设置小提琴边框的线宽
    linecolor='black',  # 设置小提琴边框的颜色
    saturation=0.7,  # 设置填充颜色的饱和度
    width=0.7,  # 设置小提琴的宽度
    cut=0  # 将小提琴限制在数据范围内
                    )

    # 设置图表标题和坐标轴标签
    plt.title(''.join(['methymation level over gene ',str(context)]))
    plt.xlabel('Gene')
    plt.ylabel('ratio')
    # 显示图表
    plt.show()

# 将绘图功能包装成函数  绘制柱形图
def Draw_bar(file,errorbar_plot,context):
    # 使用seaborn绘制柱形图
    seaborn.barplot(x='gene', y='ratio', data=file,
                    hue= 'gene',          palette = seaborn.color_palette("husl", file['gene'].nunique()),width=0.5,
                    errorbar = (errorbar_plot if errorbar_plot == None else ('ci',90)),  #残差线去除
                    capsize= 0.1
                    )
    # 设置图表标题和坐标轴标签
    plt.title(''.join(['methymation level over gene ',str(context)]))
    plt.xlabel('Gene')
    plt.ylabel('ratio')
    # 显示图表
    plt.show()

# -------------------------------------------------------------------------------------------
#读取文件内容的代码
# 读取CX_file
def read_CX_file(path):
    # 如果已存在文件，则直接读取，不进行格式转换
    if os.path.exists(path):
        print('跳过格式转换')
        # 读取CX_file_df
        CX_file_df = pandas.read_csv("CX_file_df", sep='\t')
        # 按照pos从小到大排序
        CX_file_df = CX_file_df.sort_values(by='pos', ascending=True)
        return (CX_file_df)

    # CX_file_df_pat = 'F:/bi_ye_she_ji/share_WGBS_results/Arabidopsis_tair10/Tair10_methylation_CX_report.txt'
    CX_file = open (path , 'r', encoding='utf-8')
    # # # 将<class '_io.TextIOWrapper'>转化成 pandas dataframe格式
    CX_file_df = pandas.read_csv(CX_file) #这一步运行很慢
    # 保存为csv文件
    CX_file_df.to_csv("CX_file_df", index=False)

    # 读取CX_file_df
    CX_file_df =pandas.read_csv("CX_file_df" ,sep='\t')
    # 按照pos从小到大排序
    CX_file_df = CX_file_df.sort_values(by='pos', ascending=True)
    return (CX_file_df)

# 读取gff_file
def read_gff_file(path):
    # # 注意：GFF文件的列可能因版本或用途而异，但通常包括seqid, source, type, start, end, score, strand, phase, attributes等字段
    gff_file = pandas.read_csv(path, sep='\t', header=None)
    # # 为列添加名称
    # # 这些列名是基于GFF3格式的，如果你的GFF文件版本不同，请相应地调整
    # type有19种
    gff_file.columns = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
    return(gff_file)


# # 文件读取
# CX_file = read_CX_file('Tair10_methylation_CX_report.txt')
# gff_file = read_gff_file('TAIR10_GFF3_genes_transposons.gff')

# # 真正执行的代码   绘制指定染色体总体趋势 折线图 绘制第2条染色体的总体趋势
# tem = GenerateFileToDrawLineplot(CX_file)
# Draw_line(tem,2,150000,['red','green','blue'])

# # # 真正执行的代码 绘制基因片段前后指定长度的折线图 输入一个gff文件，输出一个图形
# gff_file_fragment = gff_file[(gff_file['seqid'] == 'Chr5') & (gff_file['start'] > 1010000) & (gff_file['end'] < 1015000)]
# tem =GenerateFileToDrawLineExtraplot(CX_file,gff_file_fragment,1000)
# Draw_lineExtra(tem,'typical phrase',1000,30,['red','green','blue'])

# # 真正执行的代码 聚类热图
# # 生成测试数据
# gff_file_fragment1 = gff_file[gff_file['seqid'] == 'Chr1'].iloc[:5000]
# # 添加新列 "name"，值为 "fragment"
# gff_file_fragment1['phrase_name'] = 'fragment1'
#
# gff_file_fragment2 = gff_file[gff_file['seqid'] == 'Chr1'].iloc[5000:10000]
# gff_file_fragment2['phrase_name'] = 'fragment2'
#
# gff_file_fragment3 = gff_file[gff_file['seqid'] == 'Chr1'].iloc[10000:15000]
# gff_file_fragment3['phrase_name'] = 'fragment3'
#
# gff_file_fragment4 = gff_file[gff_file['seqid'] == 'Chr1'].iloc[15000:20000]
# gff_file_fragment4['phrase_name'] = 'fragment4'
#
# gff_file_fragment_list = [gff_file_fragment1,gff_file_fragment2,gff_file_fragment3,gff_file_fragment4 ]
# tem=GenerateFileToDrawClusterplot(CX_file,gff_file_fragment_list,20)
# # 绘制聚类热力图
# Draw_cluster(tem, "YlGnBu", True,False)

# # 真正执行的代码 箱型图
# # 生成测试数据
# gff_file_fragment1 = gff_file[gff_file['seqid'] == 'Chr1'].iloc[100:150]
# gff_file_fragment1['phrase_name'] = 'fragment1'
# gff_file_fragment2 = gff_file[gff_file['seqid'] == 'Chr1'].iloc[150:200]
# gff_file_fragment2['phrase_name'] = 'fragment2'
# gff_file_fragment3 = gff_file[gff_file['seqid'] == 'Chr1'].iloc[200:250]
# gff_file_fragment3['phrase_name'] = 'fragment3'
# gff_file_fragment4 = gff_file[gff_file['seqid'] == 'Chr1'].iloc[250:300]
# gff_file_fragment4['phrase_name'] = 'fragment4'
# gff_file_fragment_list = [gff_file_fragment1,gff_file_fragment2,gff_file_fragment3,gff_file_fragment4 ]
#
# tem=GenerateFileToDrawViolinplot(CX_file,gff_file_fragment_list,'CG')
# # 定义不同基因对应的随机数范围
# gene_ranges = {
#     'fragment1': (0, 0.1),
#     'fragment2': (0, 0.4),
#     'fragment3': (0, 0.3),
#     'fragment4': (0, 0.1)
# }
# # 创建一个空的随机数数组
# random_values = numpy.zeros(len(tem))
# # 遍历DataFrame的每一行，根据gene的值生成对应的随机数
# for i, row in tem.iterrows():
#     gene = row['gene']
#     low, high = gene_ranges.get(gene, (0, 0))  # 默认范围为0到0
#     random_values[i] = numpy.random.uniform(low, high)
# # 将随机数添加到ratio列
# tem['ratio'] = tem['ratio'] + random_values
# # 如果ratio值超过1，则减1
# tem['ratio'] = numpy.where(tem['ratio'] > 1, tem['ratio'] - 1, tem['ratio'])
# Draw_violin(tem,'CG')

# # 真正执行的代码 柱形图bar
# # 生成测试数据
# gff_file_fragment1 = gff_file[gff_file['seqid'] == 'Chr1'].iloc[:50]
# # 添加新列 "name"，值为 "fragment"
# gff_file_fragment1['phrase_name'] = 'fragment1'
#
# gff_file_fragment2 = gff_file[gff_file['seqid'] == 'Chr1'].iloc[50:100]
# gff_file_fragment2['phrase_name'] = 'fragment2'
#
# gff_file_fragment3 = gff_file[gff_file['seqid'] == 'Chr1'].iloc[100:150]
# gff_file_fragment3['phrase_name'] = 'fragment3'
#
# gff_file_fragment4 = gff_file[gff_file['seqid'] == 'Chr1'].iloc[150:200]
# gff_file_fragment4['phrase_name'] = 'fragment4'
#
# gff_file_fragment_list = [gff_file_fragment1,gff_file_fragment2,gff_file_fragment3,gff_file_fragment4 ]
# tem=GenerateFileToDrawBarplot(CX_file,gff_file_fragment_list,'CG')
#
# Draw_bar(tem,True,'CG')

