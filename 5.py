import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import tushare as ts
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 重新设计的数据获取和特征工程
class EnhancedStockData:
    """使用更有效的特征"""
    
    def __init__(self, token='XXXXXXX'): #替换成你的token
        ts.set_token(token)
        self.pro = ts.pro_api()
    
    def get_data(self, code='000001.SZ', start_date='20200101', end_date='20231231'):
        """获取数据并计算增强特征"""
        try:
            print(f"获取 {code} 数据...")
            
            # 获取日线数据
            df = ts.pro_bar(
                ts_code=code,
                adj='qfq',
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty or len(df) < 100:
                print("数据不足，使用模拟数据")
                return self._generate_enhanced_data(start_date, end_date)
            
            # 数据处理
            df = df.sort_values('trade_date')
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            
            # 计算增强特征
            df = self._calculate_enhanced_features(df)
            
            print(f"数据准备完成: {len(df)} 条")
            print(f"基准总收益: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.1f}%")
            print(f"年化波动率: {df['returns'].std()*np.sqrt(252)*100:.1f}%")
            
            return df
            
        except Exception as e:
            print(f"Tushare获取失败: {e}")
            return self._generate_enhanced_data(start_date, end_date)
    
    def _calculate_enhanced_features(self, df):
        """计算增强特征"""
        # 1. 基础特征
        df['returns'] = df['close'].pct_change()
        
        # 2. 趋势特征（多时间尺度）
        for window in [5, 10, 20, 60]:
            df[f'ma{window}'] = df['close'].rolling(window).mean()
            df[f'price_ma{window}_ratio'] = df['close'] / df[f'ma{window}'] - 1
            df[f'ma_trend_{window}'] = (df['close'] > df[f'ma{window}']).astype(float)
        
        # 3. 动量特征
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # 4. 波动率特征
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_60'] = df['returns'].rolling(60).std()
        
        # 5. 成交量特征
        df['volume_ma20'] = df['vol'].rolling(20).mean()
        df['volume_ratio'] = df['vol'] / df['volume_ma20']
        df['volume_signal'] = (df['volume_ratio'] > 1.5).astype(float)
        
        # 6. 市场宽度特征（模拟）
        df['market_breadth'] = np.random.uniform(0.3, 0.8, len(df))
        
        # 7. 波动率指标
        df['atr'] = self._calculate_atr(df)
        
        # 8. 趋势强度
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        # 9. 市场状态识别
        df = self._identify_market_regime(df)
        
        # 10. 支撑阻力特征
        df['support_resistance'] = self._calculate_support_resistance(df)
        
        # 删除缺失值
        df = df.dropna()
        
        return df
    
    def _calculate_atr(self, df, period=14):
        """计算平均真实波幅"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr / df['close']  # 标准化
    
    def _calculate_trend_strength(self, df):
        """计算趋势强度"""
        # 使用ADX思路
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = pd.concat([high - low, 
                       abs(high - close.shift()), 
                       abs(low - close.shift())], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()
        
        return adx / 100  # 归一化到[0,1]
    
    def _identify_market_regime(self, df):
        """识别市场状态"""
        df['regime'] = 0  # 0:震荡, 1:牛市, -1:熊市
        
        # 使用均线判断
        price_ma20_ratio = df['close'] / df['ma20'] - 1
        ma20_ma60_ratio = df['ma20'] / df['ma60'] - 1
        
        # 牛市条件
        bull_condition = (price_ma20_ratio > 0.05) & (ma20_ma60_ratio > 0.02)
        # 熊市条件
        bear_condition = (price_ma20_ratio < -0.05) & (ma20_ma60_ratio < -0.02)
        
        df.loc[bull_condition, 'regime'] = 1
        df.loc[bear_condition, 'regime'] = -1
        
        return df
    
    def _calculate_support_resistance(self, df, window=20):
        """计算支撑阻力位"""
        close = df['close']
        
        # 使用滚动窗口计算局部高点和低点
        resistance = close.rolling(window).max()
        support = close.rolling(window).min()
        
        # 计算当前位置
        position = (close - support) / (resistance - support + 1e-8)
        
        return position
    
    def _generate_enhanced_data(self, start_date, end_date):
        """生成增强的模拟数据"""
        print("生成增强模拟数据...")
        
        dates = pd.date_range(
            start=start_date[:4]+'-'+start_date[4:6]+'-'+start_date[6:], 
            end=end_date[:4]+'-'+end_date[4:6]+'-'+end_date[6:], 
            freq='B'
        )
        
        np.random.seed(42)
        n = len(dates)
        
        # 创建有明确趋势和周期的数据
        t = np.arange(n) / n
        
        # 主趋势
        main_trend = 0.8 * t
        
        # 周期性波动
        cycle1 = 0.1 * np.sin(2 * np.pi * t * 5)  # 短周期
        cycle2 = 0.2 * np.sin(2 * np.pi * t * 1)  # 长周期
        
        # 随机波动
        noise = np.random.randn(n) * 0.02 * (1 + t)  # 波动率随时间增加
        
        # 合成价格
        trend = main_trend + cycle1 + cycle2 + np.cumsum(noise)
        prices = 100 * np.exp(trend)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'close': prices,
            'open': prices * (1 + np.random.randn(n) * 0.01 - 0.005),
            'high': prices * (1 + np.abs(np.random.randn(n) * 0.015)),
            'low': prices * (1 - np.abs(np.random.randn(n) * 0.015)),
            'vol': np.random.randint(1000000, 20000000, n)
        }, index=dates)
        
        return self._calculate_enhanced_features(df)

# 2. 完全重构的交易环境
class RegimeAwareTradingEnv:
    """市场状态感知的交易环境"""
    
    def __init__(self, data, initial_balance=100000, transaction_cost=0.0003):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # 状态参数
        self.state_dim = 12
        self.action_dim = 3  # 0:卖出, 1:持有, 2:买入
        
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.current_step = 100  # 从第100个数据点开始
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        self.done = False
        
        # 交易记录
        self.trades = []
        self.entry_prices = []
        self.trade_durations = []
        
        # 绩效跟踪
        self.returns_history = []
        self.benchmark_returns = []
        self.max_drawdown = 0
        self.peak_value = self.initial_balance
        
        # 策略状态
        self.current_regime = 0
        self.consecutive_days = 0
        self.last_action = 1
        self.position_age = 0
        
        # 风险控制
        self.stop_loss_price = None
        self.trailing_stop = None
        
        return self._get_state()
    
    def _get_state(self):
        """获取增强状态（12维）"""
        if self.current_step >= len(self.data) - 1:
            return None
        
        row = self.data.iloc[self.current_step]
        
        # 1. 趋势特征
        trend_strength = row.get('trend_strength', 0)
        price_ma20_ratio = row.get('price_ma20_ratio', 0)
        price_ma60_ratio = row.get('price_ma60_ratio', 0)
        
        # 2. 动量特征
        momentum_5 = row.get('momentum_5', 0)
        momentum_20 = row.get('momentum_20', 0)
        
        # 3. 波动率特征
        volatility = row.get('volatility_20', 0.02) * np.sqrt(252)
        
        # 4. 成交量特征
        volume_signal = row.get('volume_signal', 0)
        
        # 5. 市场状态
        regime = row.get('regime', 0)
        
        # 6. 支撑阻力
        support_resistance = row.get('support_resistance', 0.5)
        
        # 7. 技术指标
        atr = row.get('atr', 0.02)
        
        # 8. 持仓特征
        position_ratio = (self.shares_held * row['close']) / self.total_value if self.total_value > 0 else 0
        
        # 9. 时间特征
        time_in_month = (self.current_step % 21) / 21
        
        # 10. 波动率状态
        volatility_regime = 1 if volatility > 0.25 else 0
        
        # 11. 市场宽度
        market_breadth = row.get('market_breadth', 0.5)
        
        # 12. 价格位置
        price_position = (row['close'] - self.data['close'].iloc[max(0, self.current_step-252):self.current_step].min()) / \
                        (self.data['close'].iloc[max(0, self.current_step-252):self.current_step].max() - 
                         self.data['close'].iloc[max(0, self.current_step-252):self.current_step].min() + 1e-8)
        
        state = np.array([
            trend_strength,
            price_ma20_ratio * 10,  # 放大
            price_ma60_ratio * 10,
            momentum_5 * 100,
            momentum_20 * 100,
            volatility * 10,
            volume_signal,
            regime,
            support_resistance * 2 - 1,  # 转换到[-1,1]
            atr * 100,
            position_ratio * 2 - 1,  # 转换到[-1,1]
            market_breadth * 2 - 1
        ], dtype=np.float32)
        
        # 记录当前市场状态
        self.current_regime = regime
        
        return state
    
    def step(self, action):
        """执行交易步骤"""
        if self.done:
            raise ValueError("Episode已结束")
        
        row = self.data.iloc[self.current_step]
        current_price = row['close']
        prev_value = self.total_value
        
        # 获取基准收益
        benchmark_return = row.get('returns', 0)
        self.benchmark_returns.append(benchmark_return)
        
        # 智能动作执行
        executed_action = self._intelligent_action_selection(action, row)
        
        # 执行交易
        trade_executed = False
        if executed_action == 0 and self.shares_held > 0:  # 卖出
            trade_executed = self._execute_sell(current_price, row)
        elif executed_action == 2:  # 买入
            trade_executed = self._execute_buy(current_price, row)
        
        # 更新持仓时间
        if self.shares_held > 0:
            self.position_age += 1
        else:
            self.position_age = 0
        
        # 更新资产
        self.total_value = self.balance + self.shares_held * current_price
        
        # 计算收益
        strategy_return = (self.total_value - prev_value) / prev_value if prev_value > 0 else 0
        self.returns_history.append(strategy_return)
        
        # 更新最高值和最大回撤
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
        
        current_drawdown = (self.peak_value - self.total_value) / self.peak_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # 记录动作
        self.last_action = executed_action
        
        # 移动到下一步
        self.current_step += 1
        
        # 检查是否结束
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # 获取下一状态
        next_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_enhanced_reward(
            strategy_return, 
            benchmark_return, 
            executed_action, 
            trade_executed,
            row
        )
        
        return next_state, reward, self.done, {
            'value': self.total_value,
            'strategy_return': strategy_return,
            'benchmark_return': benchmark_return,
            'action': executed_action,
            'position_ratio': (self.shares_held * current_price) / self.total_value if self.total_value > 0 else 0,
            'regime': self.current_regime
        }
    
    def _intelligent_action_selection(self, action, row):
        """智能动作选择"""
        # 基础动作
        base_action = action
        
        # 根据市场状态调整
        regime = self.current_regime
        
        # 牛市强化买入信号
        if regime > 0.5:  # 强牛市
            if base_action == 1:  # 持有改为买入
                return 2
            elif base_action == 0:  # 卖出改为持有
                return 1
        
        # 熊市强化卖出信号
        elif regime < -0.5:  # 强熊市
            if base_action == 1:  # 持有改为卖出
                return 0
            elif base_action == 2:  # 买入改为持有
                return 1
        
        # 风险控制：高波动率时谨慎
        volatility = row.get('volatility_20', 0.02) * np.sqrt(252)
        if volatility > 0.3:  # 高波动率
            if base_action == 2:  # 买入改为持有
                return 1
            elif self.position_age > 5:  # 持仓时间较长
                return 0  # 考虑卖出
        
        # 止损检查
        if self.stop_loss_price and row['close'] < self.stop_loss_price:
            return 0  # 触发止损
        
        return base_action
    
    def _execute_sell(self, price, row):
        """执行卖出"""
        if self.shares_held <= 0:
            return False
        
        sell_shares = self.shares_held
        sell_value = sell_shares * price * (1 - self.transaction_cost)
        self.balance += sell_value
        self.shares_held = 0
        
        # 记录交易
        self.trades.append({
            'step': self.current_step,
            'action': 'sell',
            'price': price,
            'shares': sell_shares,
            'reason': 'strategy'
        })
        
        # 重置止损
        self.stop_loss_price = None
        self.trailing_stop = None
        
        return True
    
    def _execute_buy(self, price, row):
        """执行买入"""
        if self.balance <= price * 100:  # 至少能买100股
            return False
        
        # 根据市场状态决定仓位
        if self.current_regime > 0.5:  # 牛市
            position_size = 0.9
        elif self.current_regime < -0.5:  # 熊市
            position_size = 0.3
        else:  # 震荡市
            position_size = 0.6
        
        # 根据波动率调整
        volatility = row.get('volatility_20', 0.02) * np.sqrt(252)
        if volatility > 0.25:  # 高波动率
            position_size *= 0.7
        
        available_cash = self.balance * position_size
        buy_shares = int(available_cash / (price * (1 + self.transaction_cost)))
        
        if buy_shares <= 0:
            return False
        
        buy_cost = buy_shares * price * (1 + self.transaction_cost)
        self.balance -= buy_cost
        self.shares_held += buy_shares
        
        # 记录交易
        self.trades.append({
            'step': self.current_step,
            'action': 'buy',
            'price': price,
            'shares': buy_shares,
            'reason': 'strategy'
        })
        
        # 设置止损
        atr = row.get('atr', 0.02) * price
        self.stop_loss_price = price - 2 * atr
        self.trailing_stop = price * 0.95
        
        return True
    
    def _calculate_enhanced_reward(self, strategy_return, benchmark_return, action, trade_executed, row):
        """计算增强的奖励函数"""
        
        # 1. 超额收益奖励（核心）
        excess_return = strategy_return - benchmark_return
        excess_reward = excess_return * 50  # 放大
        
        # 2. 风险调整奖励
        volatility = row.get('volatility_20', 0.02)
        if volatility > 0:
            sharpe_component = excess_return / volatility
        else:
            sharpe_component = 0
        
        risk_adjusted_reward = sharpe_component * 5
        
        # 3. 趋势跟随奖励
        trend_reward = 0
        trend_strength = row.get('trend_strength', 0)
        regime = self.current_regime
        
        if regime > 0 and action == 2:  # 牛市中买入
            trend_reward = 0.01 * trend_strength
        elif regime < 0 and action == 0:  # 熊市中卖出
            trend_reward = 0.01 * abs(regime)
        
        # 4. 交易效率奖励
        efficiency_reward = 0
        if trade_executed:
            # 大额交易奖励
            position_ratio = (self.shares_held * row['close']) / self.total_value if self.total_value > 0 else 0
            if position_ratio > 0.5:
                efficiency_reward = 0.005
            else:
                efficiency_reward = 0.002
            
            # 正确的市场状态交易额外奖励
            if (regime > 0 and action == 2) or (regime < 0 and action == 0):
                efficiency_reward *= 2
        
        # 5. 持仓时间惩罚（避免过久持仓）
        holding_penalty = 0
        if self.position_age > 20:  # 持仓超过20天
            holding_penalty = -0.001 * (self.position_age - 20)
        
        # 6. 交易成本惩罚
        cost_penalty = -0.002 if trade_executed else 0
        
        # 7. 回撤惩罚
        drawdown_penalty = 0
        if self.max_drawdown > 0.1:  # 回撤超过10%
            drawdown_penalty = -0.01 * (self.max_drawdown / 0.1)
        
        # 8. 连胜奖励
        win_streak_reward = 0
        if len(self.returns_history) > 5:
            recent_returns = self.returns_history[-5:]
            recent_wins = sum(1 for r in recent_returns if r > 0)
            if recent_wins >= 4:  # 最近5次赢4次
                win_streak_reward = 0.003
        
        # 总奖励
        total_reward = (
            excess_reward + 
            risk_adjusted_reward + 
            trend_reward + 
            efficiency_reward + 
            holding_penalty + 
            cost_penalty + 
            drawdown_penalty + 
            win_streak_reward
        )
        
        # 归一化
        total_reward = np.clip(total_reward, -1.0, 1.5)
        
        return float(total_reward)
    
    def get_performance_metrics(self):
        """获取性能指标"""
        if len(self.returns_history) < 2:
            return {}
        
        returns = np.array(self.returns_history)
        benchmark_returns = np.array(self.benchmark_returns)
        
        # 总收益
        total_return = (self.total_value / self.initial_balance - 1) * 100
        benchmark_total_return = (np.prod(1 + benchmark_returns) - 1) * 100
        
        # 年化收益
        n_days = len(returns)
        annual_return = ((1 + total_return/100) ** (252/n_days) - 1) * 100
        benchmark_annual = ((1 + benchmark_total_return/100) ** (252/n_days) - 1) * 100
        
        # 夏普比率
        if returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0
        
        if benchmark_returns.std() > 0:
            benchmark_sharpe = np.sqrt(252) * benchmark_returns.mean() / benchmark_returns.std()
        else:
            benchmark_sharpe = 0
        
        # 最大回撤
        values = [self.initial_balance]
        for r in returns:
            values.append(values[-1] * (1 + r))
        
        values_array = np.array(values)
        peak = np.maximum.accumulate(values_array)
        drawdown = (peak - values_array) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # 胜率
        win_rate = np.mean(returns > 0) * 100
        
        # 盈亏比
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        if len(losing_returns) > 0:
            profit_factor = abs(winning_returns.mean() / losing_returns.mean())
        else:
            profit_factor = float('inf') if len(winning_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_total_return,
            'excess_return': total_return - benchmark_total_return,
            'annual_return': annual_return,
            'benchmark_annual': benchmark_annual,
            'sharpe': sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades': len(self.trades)
        }

# 3. 增强的神经网络
class EnhancedPolicyNetwork(nn.Module):
    """增强的策略网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(EnhancedPolicyNetwork, self).__init__()
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim//4, action_dim)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim//4, 1)
        )
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # 策略输出
        policy_logits = self.policy_head(features)
        action_probs = torch.softmax(policy_logits, dim=-1)
        
        # 价值输出
        state_value = self.value_head(features)
        
        return action_probs, state_value

# 4. 增强的PPO算法
class EnhancedPPO:
    """增强的PPO算法"""
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 clip_epsilon=0.15, entropy_coef=0.01, ppo_epochs=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        
        # 策略网络
        self.policy = EnhancedPolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # 训练统计
        self.best_score = -float('inf')
        self.episode_rewards = []
    
    def select_action(self, state, explore=True, exploration_rate=0.2):
        """选择动作，带智能探索"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, state_value = self.policy(state_tensor)
            action_probs_np = action_probs.cpu().numpy()[0]
        
        # 智能探索
        if explore and np.random.random() < exploration_rate:
            # 根据概率分布进行探索，但偏向高概率动作
            probs = action_probs_np
            probs = probs ** 2  # 让高概率动作更高
            probs = probs / probs.sum()
            action = np.random.choice(self.action_dim, p=probs)
        else:
            # 利用：选择最优动作
            action = np.argmax(action_probs_np)
        
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(torch.tensor([action]))
        
        return action, log_prob.item(), state_value.item()
    
    def store_transition(self, state, action, log_prob, value, reward, done):
        """存储转移"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self, next_value, episode_reward):
        """PPO更新"""
        if len(self.states) < 16:  # 最小批量
            self.clear_memory()
            return
        
        # 记录奖励
        self.episode_rewards.append(episode_reward)
        
        # 计算GAE
        returns, advantages = self._compute_gae(next_value)
        
        # 转换为tensor
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # 多轮PPO更新
        for epoch in range(self.ppo_epochs):
            # 随机采样
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), 32):  # 小批量
                end = min(start + 32, len(states))
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                action_probs, state_values = self.policy(batch_states)
                dist = Categorical(action_probs)
                
                # 计算损失
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 策略损失
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = 0.5 * nn.SmoothL1Loss()(state_values.squeeze(), batch_returns)
                
                # 熵正则化
                entropy_loss = -entropy
                
                # 总损失
                loss = policy_loss + value_loss + self.entropy_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        # 学习率调度
        self.scheduler.step()
        
        # 清空内存
        self.clear_memory()
    
    def _compute_gae(self, next_value):
        """计算GAE"""
        advantages = []
        returns = []
        
        R = next_value
        gae = 0
        gamma = self.gamma
        lam = 0.95
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 标准化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def clear_memory(self):
        """清空经验缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def save_model(self, path='enhanced_model.pth'):
        """保存模型"""
        torch.save(self.policy.state_dict(), path)
    
    def load_model(self, path='enhanced_model.pth'):
        """加载模型"""
        self.policy.load_state_dict(torch.load(path))

# 5. 增强的训练器
class EnhancedTrainer:
    """增强的训练器"""
    
    def __init__(self, env, agent, episodes=120, max_steps=400):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_steps = max_steps
        self.results = []
        self.best_excess_return = -float('inf')
    
    def train(self):
        """增强的训练过程"""
        print("开始增强PPO训练...")
        print("=" * 70)
        
        for episode in range(self.episodes):
            state = self.env.reset()
            if state is None:
                continue
            
            episode_reward = 0
            episode_values = []
            step_count = 0
            
            # 动态探索率
            exploration_rate = max(0.1, 0.3 * (1 - episode / (self.episodes * 0.7)))
            
            for step in range(self.max_steps):
                if state is None:
                    break
                
                # 选择动作
                action, log_prob, value = self.agent.select_action(
                    state, 
                    explore=(episode < self.episodes * 0.8),
                    exploration_rate=exploration_rate
                )
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                self.agent.store_transition(state, action, log_prob, value, reward, done)
                
                # 更新
                state = next_state
                episode_reward += reward
                episode_values.append(info['value'])
                
                step_count += 1
                
                if done or step == self.max_steps - 1:
                    # 计算最终值
                    if next_state is not None:
                        with torch.no_grad():
                            _, next_value = self.agent.policy(torch.FloatTensor(next_state).unsqueeze(0))
                            next_value = next_value.item()
                    else:
                        next_value = 0
                    
                    # 更新策略
                    if len(self.agent.states) > 0:
                        self.agent.update(next_value, episode_reward)
                    break
            
            # 记录结果
            if episode_values:
                final_value = episode_values[-1]
                total_return = (final_value / self.env.initial_balance - 1) * 100
                
                # 获取基准收益
                metrics = self.env.get_performance_metrics()
                if metrics:
                    excess_return = metrics.get('excess_return', 0)
                    benchmark_return = metrics.get('benchmark_return', 0)
                    
                    self.results.append({
                        'episode': episode,
                        'total_return': total_return,
                        'benchmark_return': benchmark_return,
                        'excess_return': excess_return,
                        'reward': episode_reward,
                        'sharpe': metrics.get('sharpe', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'trades': metrics.get('trades', 0)
                    })
                    
                    # 保存最佳模型
                    if excess_return > self.best_excess_return:
                        self.best_excess_return = excess_return
                        self.agent.save_model('best_enhanced_model.pth')
                        
                        if excess_return > 0:
                            print(f"🎯 发现新的最佳模型！超额收益: {excess_return:.2f}%")
                    
                    # 定期输出
                    if (episode + 1) % 10 == 0 or episode == 0:
                        current_lr = self.agent.optimizer.param_groups[0]['lr']
                        
                        print(f"Episode {episode+1:3d}/{self.episodes} | "
                              f"策略: {total_return:6.2f}% | "
                              f"基准: {benchmark_return:6.2f}% | "
                              f"超额: {excess_return:6.2f}% | "
                              f"夏普: {metrics.get('sharpe', 0):5.2f} | "
                              f"回撤: {metrics.get('max_drawdown', 0):5.1f}% | "
                              f"交易: {metrics.get('trades', 0):3d}")
        
        print("=" * 70)
        print("训练完成！")
        
        return pd.DataFrame(self.results)
    
    def backtest(self, test_data):
        """增强的回测"""
        print("\n开始增强回测...")
        
        # 加载最佳模型
        try:
            self.agent.load_model('best_enhanced_model.pth')
            print("加载最佳模型")
        except:
            print("使用当前模型")
        
        # 创建测试环境
        test_env = RegimeAwareTradingEnv(
            test_data,
            self.env.initial_balance,
            self.env.transaction_cost
        )
        
        state = test_env.reset()
        test_values = [test_env.initial_balance]
        test_actions = []
        test_prices = []
        test_regimes = []
        
        while True:
            if state is None:
                break
            
            # 记录当前价格
            current_price = test_env.data.iloc[test_env.current_step]['close']
            test_prices.append(current_price)
            
            # 选择动作（无探索）
            with torch.no_grad():
                action_probs, _ = self.agent.policy(torch.FloatTensor(state).unsqueeze(0))
                action = torch.argmax(action_probs).item()
            
            # 执行动作
            next_state, _, done, info = test_env.step(action)
            
            # 记录
            test_values.append(info['value'])
            test_actions.append(info['action'])
            test_regimes.append(info.get('regime', 0))
            
            state = next_state
            
            if done:
                break
        
        # 性能分析
        self._analyze_performance(test_values, test_prices, test_actions, test_regimes, test_env)
        
        return test_values, test_actions, test_prices, test_regimes
    
    def _analyze_performance(self, values, prices, actions, regimes, env):
        """详细性能分析"""
        if len(values) < 2 or len(prices) < 2:
            return
        
        # 获取指标
        metrics = env.get_performance_metrics()
        
        if not metrics:
            return
        
        print(f"\n{'='*70}")
        print(f"{'增强回测结果':^70}")
        print(f"{'='*70}")
        
        # 基础指标
        print(f"{'指标':<25} {'策略':>15} {'基准':>15} {'超额':>15}")
        print(f"{'-'*70}")
        
        key_metrics = [
            ('总收益率', metrics['total_return'], metrics['benchmark_return'], metrics['excess_return']),
            ('年化收益率', metrics['annual_return'], metrics['benchmark_annual'], metrics['annual_return'] - metrics['benchmark_annual']),
            ('夏普比率', metrics['sharpe'], metrics['benchmark_sharpe'], metrics['sharpe'] - metrics['benchmark_sharpe']),
            ('最大回撤', metrics['max_drawdown'], '', ''),
            ('胜率', metrics['win_rate'], '', ''),
            ('盈亏比', metrics['profit_factor'], '', ''),
            ('交易次数', metrics['trades'], '', ''),
        ]
        
        for name, strategy, bench, excess in key_metrics:
            if isinstance(strategy, float):
                strategy_fmt = f"{strategy:>15.2f}{'%' if name != '夏普比率' and name != '盈亏比' else ''}"
                bench_fmt = f"{bench:>15.2f}{'%' if bench != '' and name != '夏普比率' else ''}" if bench != '' else " " * 15
                excess_fmt = f"{excess:>+15.2f}{'%' if excess != '' and name != '夏普比率' else ''}" if excess != '' else " " * 15
            else:
                strategy_fmt = f"{strategy:>15}"
                bench_fmt = f"{bench:>15}" if bench != '' else " " * 15
                excess_fmt = f"{excess:>15}" if excess != '' else " " * 15
            
            print(f"{name:<25} {strategy_fmt} {bench_fmt} {excess_fmt}")
        
        print(f"{'='*70}")
        
        # 市场状态分析
        if regimes:
            bull_days = sum(1 for r in regimes if r > 0.5)
            bear_days = sum(1 for r in regimes if r < -0.5)
            neutral_days = len(regimes) - bull_days - bear_days
            
            print(f"\n市场状态分析:")
            print(f"  牛市天数: {bull_days} ({bull_days/len(regimes)*100:.1f}%)")
            print(f"  熊市天数: {bear_days} ({bear_days/len(regimes)*100:.1f}%)")
            print(f"  震荡天数: {neutral_days} ({neutral_days/len(regimes)*100:.1f}%)")
        
        # 动作分析
        action_counts = {0: 0, 1: 0, 2: 0}
        for a in actions:
            if a in action_counts:
                action_counts[a] += 1
        
        print(f"\n动作分布:")
        print(f"  卖出: {action_counts[0]} ({action_counts[0]/len(actions)*100:.1f}%)")
        print(f"  持有: {action_counts[1]} ({action_counts[1]/len(actions)*100:.1f}%)")
        print(f"  买入: {action_counts[2]} ({action_counts[2]/len(actions)*100:.1f}%)")
        
        # 交易分析
        if env.trades:
            buy_trades = [t for t in env.trades if t['action'] == 'buy']
            sell_trades = [t for t in env.trades if t['action'] == 'sell']
            
            print(f"\n交易分析:")
            print(f"  买入次数: {len(buy_trades)}")
            print(f"  卖出次数: {len(sell_trades)}")
            
            if buy_trades and sell_trades:
                avg_buy_price = np.mean([t['price'] for t in buy_trades])
                avg_sell_price = np.mean([t['price'] for t in sell_trades])
                avg_return = (avg_sell_price / avg_buy_price - 1) * 100
                print(f"  平均买卖收益率: {avg_return:.2f}%")
        
        # 结论
        excess = metrics['excess_return']
        if excess > 5:
            print(f"\n🎉 策略大幅跑赢基准！超额收益: {excess:.2f}%")
        elif excess > 0:
            print(f"\n✅ 策略跑赢基准，超额收益: {excess:.2f}%")
        elif excess > -2:
            print(f"\n⚠️  策略与基准接近，差异: {excess:.2f}%")
        else:
            print(f"\n❌ 策略未跑赢基准，落后: {abs(excess):.2f}%")

# 6. 增强的可视化
def visualize_enhanced_results(train_results, test_values, test_prices, test_actions, test_regimes):
    """增强的可视化"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. 训练超额收益
    ax1 = axes[0, 0]
    ax1.plot(train_results['episode'], train_results['excess_return'], 
             linewidth=2, color='blue', label='超额收益')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.fill_between(train_results['episode'], 0, train_results['excess_return'], 
                     where=train_results['excess_return']>0, color='green', alpha=0.3)
    ax1.fill_between(train_results['episode'], 0, train_results['excess_return'], 
                     where=train_results['excess_return']<0, color='red', alpha=0.3)
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('超额收益 (%)')
    ax1.set_title('训练超额收益曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 收益对比
    ax2 = axes[0, 1]
    episodes = train_results['episode']
    ax2.plot(episodes, train_results['total_return'], 
             label='策略收益', linewidth=2, color='blue')
    ax2.plot(episodes, train_results['benchmark_return'], 
             label='基准收益', linewidth=2, color='red', alpha=0.7)
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('收益率 (%)')
    ax2.set_title('策略 vs 基准收益')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 夏普比率
    ax3 = axes[0, 2]
    ax3.plot(episodes, train_results['sharpe'], 
             linewidth=2, color='green', label='夏普比率')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('夏普比率')
    ax3.set_title('训练夏普比率')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 净值曲线对比
    ax4 = axes[1, 0]
    strategy_norm = [v/test_values[0] for v in test_values]
    benchmark_norm = [p/test_prices[0] for p in test_prices]
    
    # 计算收益
    strategy_return = (test_values[-1]/test_values[0]-1)*100
    benchmark_return = (test_prices[-1]/test_prices[0]-1)*100
    
    ax4.plot(strategy_norm, label=f'策略 ({strategy_return:.1f}%)', 
             linewidth=2.5, color='blue')
    ax4.plot(benchmark_norm, label=f'基准 ({benchmark_return:.1f}%)', 
             linewidth=2, color='red', alpha=0.7, linestyle='--')
    ax4.set_xlabel('时间')
    ax4.set_ylabel('归一化净值')
    ax4.set_title('净值曲线对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 累计收益对比
    ax5 = axes[1, 1]
    strategy_cum = np.cumprod([1 + r for r in np.diff(test_values)/test_values[:-1]])
    benchmark_cum = np.cumprod([1 + r for r in np.diff(test_prices)/test_prices[:-1]])
    
    ax5.plot(strategy_cum, label='策略累计收益', linewidth=2, color='blue')
    ax5.plot(benchmark_cum, label='基准累计收益', linewidth=2, color='red', alpha=0.7)
    ax5.set_xlabel('时间')
    ax5.set_ylabel('累计收益')
    ax5.set_title('累计收益对比')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 市场状态与买卖信号
    ax6 = axes[1, 2]
    ax6.plot(test_prices, label='股价', linewidth=1, color='black', alpha=0.7)
    
    # 市场状态背景
    if test_regimes:
        for i in range(len(test_regimes)-1):
            if test_regimes[i] > 0.5:  # 牛市
                ax6.axvspan(i, i+1, alpha=0.1, color='green')
            elif test_regimes[i] < -0.5:  # 熊市
                ax6.axvspan(i, i+1, alpha=0.1, color='red')
    
    # 买卖信号
    buy_idx = [i for i, a in enumerate(test_actions) if a == 2 and i < len(test_prices)]
    sell_idx = [i for i, a in enumerate(test_actions) if a == 0 and i < len(test_prices)]
    
    if buy_idx:
        ax6.scatter(buy_idx, [test_prices[i] for i in buy_idx], 
                   color='green', s=80, marker='^', label='买入', zorder=5, alpha=0.9)
    if sell_idx:
        ax6.scatter(sell_idx, [test_prices[i] for i in sell_idx], 
                   color='red', s=80, marker='v', label='卖出', zorder=5, alpha=0.9)
    
    ax6.set_xlabel('时间')
    ax6.set_ylabel('价格')
    ax6.set_title('市场状态与买卖信号')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 回撤曲线
    ax7 = axes[2, 0]
    values_array = np.array(test_values)
    peak = np.maximum.accumulate(values_array)
    drawdown = (peak - values_array) / peak
    
    ax7.fill_between(range(len(drawdown)), 0, drawdown*100, 
                     color='red', alpha=0.3, label='回撤')
    ax7.plot(drawdown*100, linewidth=1, color='red', alpha=0.7)
    ax7.set_xlabel('时间')
    ax7.set_ylabel('回撤 (%)')
    ax7.set_title('回撤曲线')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 动作分布
    ax8 = axes[2, 1]
    action_counts = {0: 0, 1: 0, 2: 0}
    for a in test_actions:
        if a in action_counts:
            action_counts[a] += 1
    
    labels = ['卖出', '持有', '买入']
    counts = [action_counts[0], action_counts[1], action_counts[2]]
    colors = ['red', 'gray', 'green']
    
    bars = ax8.bar(labels, counts, color=colors, alpha=0.7)
    ax8.set_xlabel('动作')
    ax8.set_ylabel('次数')
    ax8.set_title('动作分布')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{count}\n({count/len(test_actions)*100:.1f}%)', 
                ha='center', va='bottom', fontsize=9)
    
    # 9. 收益率分布
    ax9 = axes[2, 2]
    if len(test_values) > 1:
        strategy_returns = np.diff(test_values) / test_values[:-1]
        benchmark_returns = np.diff(test_prices) / test_prices[:-1]
        
        ax9.hist(strategy_returns*100, bins=30, alpha=0.6, color='blue', 
                label=f'策略 (均值={strategy_returns.mean()*100:.3f}%)', density=True)
        ax9.hist(benchmark_returns*100, bins=30, alpha=0.6, color='red', 
                label=f'基准 (均值={benchmark_returns.mean()*100:.3f}%)', density=True)
        
        ax9.axvline(x=strategy_returns.mean()*100, color='blue', linestyle='--', alpha=0.7)
        ax9.axvline(x=benchmark_returns.mean()*100, color='red', linestyle='--', alpha=0.7)
        
        ax9.set_xlabel('日收益率 (%)')
        ax9.set_ylabel('频率')
        ax9.set_title('收益率分布')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_trading_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 7. 主程序
def main():
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("🚀 开始市场状态感知PPO策略训练")
    print("=" * 70)
    
    # 1. 获取数据
    data_loader = EnhancedStockData()  # 替换Token
    
    data = data_loader.get_data(
        code='000333.SZ',  # 平安银行
        start_date='20190101',
        end_date='20260501'
    )
    
    if data.empty or len(data) < 200:
        print("数据不足，退出")
        return
    
    # 2. 划分数据
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"\n📊 数据划分:")
    print(f"训练集: {len(train_data)} 条 ({train_data.index[0].date()} 到 {train_data.index[-1].date()})")
    print(f"测试集: {len(test_data)} 条 ({test_data.index[0].date()} 到 {test_data.index[-1].date()})")
    
    # 3. 创建环境
    env = RegimeAwareTradingEnv(train_data, initial_balance=100000, transaction_cost=0.0003)
    state = env.reset()
    
    print(f"\n🤖 模型配置:")
    print(f"状态维度: {env.state_dim}")
    print(f"动作空间: {env.action_dim} (0:卖出, 1:持有, 2:买入)")
    
    # 4. 创建智能体
    agent = EnhancedPPO(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=1e-3,
        gamma=0.9,
        clip_epsilon=0.15
    )
    
    # 5. 训练
    trainer = EnhancedTrainer(env, agent, episodes=200, max_steps=2300)
    train_results = trainer.train()
    
    # 6. 回测
    test_values, test_actions, test_prices, test_regimes = trainer.backtest(test_data)
    
    if len(test_values) < 2 or len(test_prices) < 2:
        print("回测数据不足")
        return
    
    # 7. 可视化
    visualize_enhanced_results(train_results, test_values, test_prices, test_actions, test_regimes)
    
    return agent, train_results, test_values

# 运行
if __name__ == "__main__":
    # 注意：替换YOUR_TUSHARE_TOKEN为你的真实Token
    agent, train_results, test_values = main()