from jqdata import *
import pandas as pd
import numpy as np


# 代码须在聚宽回测环境下运行
# 策略思路来自研报"东方证券-《因子选股系列研究之六十二》：来自优秀基金经理的超额收益"的研究结果
# 策略逻辑见my_trade()

# 聚宽回测初始化函数
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 避免未来数据
    set_option("avoid_future_data", True)

    # 输出内容到日志 log.info()
    log.info('初始函数开始运行且全局只运行一次')
    # 过滤掉order系列API产生的比error级别低的log
    log.set_level('order', 'error')

    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
                   type='stock')

    # 每月第一个交易日，开盘时运行
    run_monthly(my_trade, 1, time='open', reference_security='000300.XSHG')


def my_trade(context):
    """跟随优质基金的配置股票

    策略思路：
    1. 筛选出1亿~100亿规模的股票型或混合型 开放式基金
    2. 获取基金数据，计算夏普率，比提取出最高的前10%
    3. 获取前10的持仓股票，根据仓位占比权重排名，选出前10%的股票
    4. 平均持仓股票，以获得超额收益

    当前回测时间为季报公布的次月时，运行
    反之，直接返回
    每季运行一次，每季调整一次仓位

    Args:
        context: context对象，包含回测信息总览，账户、时间等
                 详细：https://image.joinquant.com/47ef029328c901270310864ff17dbc09
    """
    # 回测时的当前时间
    curr_dt = context.current_dt.date()

    # 仅在季报公布次月运行
    if curr_dt.month not in [2, 5, 8, 11]:
        return

    # 根据优秀基金持仓计算：应持仓的股票和仓位
    fund_data = ProcessFundData(curr_dt)
    stocks = fund_data.get_stocks()

    # 交易
    trader = Trader(context, stocks=stocks)
    trader.rebalance()
    return


# 获取基金的数据
class QueryFundData:
    """查询基金数据类

    Attributes：
        all_fund_info: 所有股票型或混合型的开放性基金的代码及发行时间
        date: 上个季度最后一天，用于获取信息的截止日
    """

    def __init__(self, curr_dt):
        """
        设定日期为上个季度末
        收集所有符合条件的基金代码

        Args:
            curr_dt(datetime): 回测当前日期

        """
        self.date = curr_dt - pd.offsets.QuarterEnd()

        # 收集所有所需基金代码，只运行一次
        if not hasattr(self, 'all_fund_info'):
            self.all_fund_info = self.get_all_fund_info()

        return

    # 收集所有股票型或混合型的开放性基金的基础信息
    @classmethod
    def get_all_fund_info(cls) -> pd.DataFrame:
        """获取至今所有股票型、混合型 开放式基金代码及起始日

        聚宽查询单次返回不能超过3000条，所以需要拼接
        将时间分为2017年前，和2017年后两部分，防止超过3000

        Returns:
            包含基金代码和发行日的 df
        """
        dates = [
            pd.Timestamp('1990-1-1'),
            pd.Timestamp('2017-1-1'),
            pd.Timestamp('2099-1-1'),
        ]

        # 查询基金主题信息
        def query_fund_main_info(beg_dt, end_dt) -> pd.DataFrame:
            q = query(
                finance.FUND_MAIN_INFO.start_date,
                finance.FUND_MAIN_INFO.main_code,
            ).filter(
                # 开放式基金
                finance.FUND_MAIN_INFO.operate_mode_id == '401001',
                finance.FUND_MAIN_INFO.underlying_asset_type_id.in_(
                    ['402001', '402004']  # 股票型，混合型
                ),
                finance.FUND_MAIN_INFO.start_date >= beg_dt,
                finance.FUND_MAIN_INFO.start_date < end_dt,
            )

            return finance.run_query(q)

        # 拼接df
        df_list = map(query_fund_main_info, dates[:-1], dates[1:])
        df = pd.concat(df_list, ignore_index=True)

        # 修改列名，修改时间格式
        df.rename(columns={'main_code': 'code'}, inplace=True)
        df.start_date = pd.to_datetime(df.start_date)
        return df

    # 收集给定基金的总资产，限定总资产在 1亿~100亿范围内
    def get_fund_total_asset(self, fund_codes) -> pd.DataFrame:
        """获得基金的上一季度末的总资产

        因为基金数可能超过3000，聚宽返回不超过3000条
        所以改为将基金分为2分，每次分别获取2000+，最后拼接

        Args:
            fund_codes(np.ndarray): 基金代码

        Returns:
            包含基金代码、当季结束日、总资产的 df
        """
        # 拆分基金列表
        num = len(fund_codes)
        fund_codes_list = np.array_split(fund_codes, num // 3000 + 1)

        # 查询基金资产组合概况，筛选出总资产在 1亿~100亿 的基金
        def query_fund_portfolio(codes) -> pd.DataFrame:
            q = query(
                finance.FUND_PORTFOLIO.code,
                finance.FUND_PORTFOLIO.period_end,
                finance.FUND_PORTFOLIO.total_asset,
            ).filter(
                finance.FUND_PORTFOLIO.total_asset > 1e8,
                finance.FUND_PORTFOLIO.total_asset < 1e100,
                finance.FUND_PORTFOLIO.code.in_(codes),
                finance.FUND_PORTFOLIO.period_end == self.date,
            )
            return finance.run_query(q)

        df_list = []

        for c in fund_codes_list:
            df_list.append(query_fund_portfolio(c))

        df = pd.concat(df_list, ignore_index=True)

        return df

    # 收集给定基金近一年的净值信息
    def get_fund_net_value(self, fund_codes) -> pd.DataFrame:
        """获取基金最近一年净值信息

        聚宽数据提取限制，需要拆分
        每年大概250个交易日，所以每次提取只12只基金的一年数据
        分批读取后合并

        Args:
            fund_codes(np.ndarray): 基金代码

        Returns:
            包含基金代码、日期、当日净值的 df
        """

        # 以上个季度最后一天为结束日，向前退1年为起始日
        period_end_date = self.date
        period_begin_date = self.date - pd.DateOffset(years=1)

        # 分成每次12只的list
        num = len(fund_codes)
        fund_codes_list = np.array_split(fund_codes, num // 12 + 1)

        # 查询基金累计复权净值
        def query_fund_net_value(codes) -> pd.DataFrame:
            q = query(
                finance.FUND_NET_VALUE.code,
                finance.FUND_NET_VALUE.day,
                finance.FUND_NET_VALUE.refactor_net_value,
            ).filter(
                finance.FUND_NET_VALUE.code.in_(codes),
                finance.FUND_NET_VALUE.day >= period_begin_date,
                finance.FUND_NET_VALUE.day < period_end_date,
            )
            return finance.run_query(q)

        df_list = map(query_fund_net_value, fund_codes_list)

        df = pd.concat(df_list, ignore_index=True)

        return df

    # 收集给定基金的股票持仓信息
    def get_fund_portfolio_stock(self, fund_codes) -> pd.DataFrame:
        """
        获得基金上季度末的前10股票仓位
        占比仓位至少 >1

        Returns:
            基金的前10股票仓位信息的df
        """

        def query_fund_portfolio_stock(codes):
            q = query(
                finance.FUND_PORTFOLIO_STOCK.code,  # 基金代码
                finance.FUND_PORTFOLIO_STOCK.name,  # 股票名称
                finance.FUND_PORTFOLIO_STOCK.symbol,  # 股票代码
                finance.FUND_PORTFOLIO_STOCK.proportion,  # 股票仓位
            ).filter(
                finance.FUND_PORTFOLIO_STOCK.code.in_(codes),
                finance.FUND_PORTFOLIO_STOCK.proportion > 1,
                finance.FUND_PORTFOLIO_STOCK.rank <= 10,
                finance.FUND_PORTFOLIO_STOCK.report_type.like('%季度'),
                finance.FUND_PORTFOLIO_STOCK.period_end == self.date,
            )
            return finance.run_query(q)

        # 若基金超过300，则分批查询
        num = len(fund_codes)
        fund_codes_list = np.array_split(fund_codes, num // 300 + 1)

        df_list = []

        for c in fund_codes_list:
            df_list.append(query_fund_portfolio_stock(c))

        df = pd.concat(df_list, ignore_index=True)

        return df


# 处理基金数据，筛选出优秀基金，根据其持仓，得出对应股票和持仓
class ProcessFundData(QueryFundData):
    """处理筛选数据类
    
    筛选出优秀基金，查询其股票持仓
    按权重排名，取前10%股票
    """

    def __init__(self, curr_dt):
        """收集信息，筛选出优秀基金

        Args:
            curr_dt(datetime): 回测当前日期
        """
        super(ProcessFundData, self).__init__(curr_dt)

        # 从基金列表中，获取已发行一年以上基金代码
        fund_codes = self.new_fund_filter()

        # 筛选出总资产在 1亿~100亿 的基金
        fund_total_asset = self.get_fund_total_asset(fund_codes)
        fund_codes = fund_total_asset.code.values

        # 获取基金近一年净值，并据此计算夏普率
        self.fund_net_value = self.get_fund_net_value(fund_codes)
        fund_sharpe_ratio = self.get_fund_sharpe_ratio()

        # 取夏普率最高的前10%基金
        sharpe_ratio_cut = pd.qcut(fund_sharpe_ratio, 10, labels=np.arange(10), )
        fund_codes = sharpe_ratio_cut[sharpe_ratio_cut == 9].index.values

        # 获取优质基金的持股信息，清洗、整合信息
        fund_portfolio_stock = self.get_fund_portfolio_stock(fund_codes)
        self.fund_portfolio_stock = self.stock_filter(fund_portfolio_stock)

        return

    # 从所有基金中筛选出已发行一年以上的
    def new_fund_filter(self) -> np.ndarray:

        last_year = self.date - pd.DateOffset(years=1)
        fund_info = self.all_fund_info

        fund_codes = fund_info.query('start_date < @last_year').code.values
        return fund_codes

    # 获得基金近一年夏普率
    def get_fund_sharpe_ratio(self) -> pd.DataFrame:
        """计算基金近一年夏普率

        Returns:
            返回包含代码、截止日、夏普率的df
        """
        fnv = self.fund_net_value.groupby('code')

        # 计算夏普率
        def calculate_sharpe_ratio(df):
            # 有些基金数据不足一年，不纳入
            if df.shape[0] < 200:
                return np.NaN

            ret = df['refactor_net_value'].pct_change()
            if ret.std() == 0:
                return np.NaN
            else:
                return ret.mean() / ret.std() * np.sqrt(252)

        fund_sharpe_ratio = fnv.apply(calculate_sharpe_ratio).dropna()
        return fund_sharpe_ratio

    # 清洗数据，并标准化股票代码
    def stock_filter(self, df) -> pd.DataFrame:
        """
        标准化股票代码
        去除st、非A股、银行股、非银金融股

        Args:
            df(pd.DataFrame): 基金持仓df

        Returns:
            处理过的基金持仓df
        """
        # 去除st
        df = df[~df.name.str.contains('ST')].copy()

        def stock_filter(*args):
            """ 去除非A股并标准化股票代码

            normalize_code(): 将其他形式的股票代码转换为聚宽可用的股票代码形式
            e.g: 000001 -> 000001.XSHE
            非A股代码会报错，返回NaN
            """
            try:
                return normalize_code(*args)
            except:
                return np.NaN

        df.symbol = df.symbol.apply(stock_filter)
        df.dropna(inplace=True)

        # 获取申万一级银行股和非银金融股
        bank_stock = get_industry_stocks('801780', self.date)
        non_bank_stock = get_industry_stocks('801790', self.date)

        # 去除银行股和非银金融股
        df = df[~df.symbol.isin(bank_stock)]
        df = df[~df.symbol.isin(non_bank_stock)]

        return df

    # 获取股票
    def get_stocks(self):
        # 股票被多只基金持仓时，取比例的之和作为股票排名权重
        fund_portfolio_stock = self.fund_portfolio_stock.groupby('symbol')
        portfolio_stock = fund_portfolio_stock['proportion'].sum()

        # 取前10%的股票
        portfolio_cut = pd.qcut(portfolio_stock, 10, labels=np.arange(10))
        stock_codes = portfolio_cut[portfolio_cut == 9].index.values

        return stock_codes


class Trader():
    """交易类

    传入股票或持仓比例，进行仓位调整

    Attributes:
        context:    聚宽context对象
        weights:    仓位比例
        stocks:     股票代码
    """

    def __init__(self, context, weights=None, stocks=None):
        """
        weights不为空，则按weights的比例进行调仓
        否则平均持仓stocks中的股票

        Args:
            context(context):   聚宽context对象
            weights(dict):      股票持仓比例
            stocks(np.ndarray):      股票代码
        """
        self.context = context

        # 持仓比重
        if weights is None:
            self.weights = dict()
        else:
            self.weights = weights

        # 股票代码
        if stocks is None:
            self.stocks = []
        else:
            self.stocks = stocks
        return

    # 记录股票持仓数变化
    @staticmethod
    def stock_logger(stock, change):
        log.info(
            f'{stock}: {change:+.0f}股'
        )

    # 换仓调仓
    def rebalance(self):
        """
        检查持仓比例是否标准
        根据股票最新价格计算持仓数量
        按数量调仓交易
        """

        # 持仓比例检查
        self.weights_check()

        # 计算持仓数量
        stock_amount: dict = self.calculate_stock_amount()

        # 账户持仓情况
        my_pos: dict = self.context.portfolio.positions

        # 减仓、清仓
        for stock in my_pos:
            change = stock_amount.get(stock, 0) - my_pos[stock].total_amount
            if change < 0:
                # 记录前后持仓数变化
                self.stock_logger(stock, change)

                # 下单
                order_target(stock, stock_amount.get(stock, 0))

        # 加仓
        for stock in my_pos:
            change = stock_amount.get(stock, 0) - my_pos[stock].total_amount
            if change > 0:
                # 记录前后持仓数变化
                self.stock_logger(stock, change)

                # 下单
                order_target(stock, stock_amount.get(stock, 0))

        # 买入
        for stock in stock_amount:
            # 跳过已交易品种
            if stock in my_pos:
                continue

            # 记录前后持仓数变化
            self.stock_logger(stock, stock_amount[stock])

            # 下单
            order_target(stock, stock_amount[stock])

        return

    # 平均持股
    def avg_position(self):
        assert len(self.stocks), '仓位比例和股票不能都为为空'

        # 根据股票数，计算持仓比例
        num = len(self.stocks)
        self.weights = {s: 1 / num for s in self.stocks}
        return

    # 持仓比例检查
    def weights_check(self):
        assert isinstance(self.weights, dict), '仓位比重必须是dict型'

        # 没有持仓比重，就平均持股
        if not self.weights:
            self.avg_position()
            return

        # 若比例之和不为 1，重新计算转化为百分比
        values_sum = sum([self.weights[s] for s in self.weights])
        if values_sum != 1.0:
            self.weights = {
                s: self.weights[s] / values_sum for s in self.weights
            }
        return

    # 根据比例，计算持仓数量
    def calculate_stock_amount(self) -> dict:
        stock_amount = dict()

        # 账户总资产，
        total_value: float = self.context.portfolio.total_value

        # 聚宽获取当前时间的所有股票数据函数
        curr_data = get_current_data()

        # 根据新的持仓比重，计算目标持股数，按最低100取整
        for stock in self.weights:
            curr_value = self.weights.get(stock, 0) * total_value
            curr_amount = curr_value // curr_data[stock].last_price
            curr_amount = curr_amount - curr_amount % 100
            stock_amount[stock] = curr_amount

        return stock_amount
