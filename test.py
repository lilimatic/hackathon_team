# region imports
from AlgorithmImports import *
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from scipy.optimize import minimize
from pypfopt import EfficientFrontier, objective_functions
# endregion

class BlackLittermanPortfolio(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2019, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        self.AddEquity("SPY", Resolution.Daily)
        self.SetBenchmark("SPY")

        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)

        self.SetPortfolioConstruction(BlackLittermanOptimizationPortfolioConstructionModel())

        self.SetRiskManagement(
            CompositeRiskManagementModel(
                MaximumDrawdownPercentPerSecurity(0.15),
                TrailingStopRiskManagementModel(0.15)
            )
        )

        self.SetExecution(ImmediateExecutionModel())
        
        self.targets = []
        self.rebalance_months = {1, 4, 7, 10}

    #top 10 S&P500
    def CoarseSelectionFunction(self, coarse):
        sortedByDollarVolume = sorted([x for x in coarse if x.HasFundamentalData], key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sortedByDollarVolume[:10]]

    def CheckRebalance(self):
        if self.Time.month in self.rebalance_months:
            self.Rebalance()

    def Rebalance(self):
        if len(self.targets) > 0:
            self.SetHoldings(self.targets)
            self.targets.clear()
    
    def OnData(self, data):
        if not self.Portfolio.Invested:
            prices = self.History([x.Symbol for x in self.ActiveSecurities.Values], 252, Resolution.Daily)["close"].unstack(level=0)

            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

            market_prices = self.History(["SPY"], 252, Resolution.Daily)["close"]["SPY"]
            delta = black_litterman.market_implied_risk_aversion(market_prices.pct_change().dropna())

            market_prior = prices.pct_change().mean()
            P, Q = self.GenerateDynamicViews(prices)
            bl = BlackLittermanModel(S, pi=market_prior, risk_aversion=delta, Q=Q, P=P)
            bl_returns = bl.bl_returns()

            weights = self.OptimizePortfolio(bl_returns, S)

            self.targets = [PortfolioTarget(symbol, weight) for symbol, weight in weights.items()]
    
    def GenerateDynamicViews(self, prices):
        pass

        return P, Q

    def OptimizePortfolio(self, bl_returns, cov_matrix):
        ef = EfficientFrontier(bl_returns, cov_matrix)
        ef.add_objective(objective_functions.L2_reg)
        ef.max_sharpe()
        weights = ef.clean_weights()
        return weights

    # def OnSecuritiesChanged(self, changes):
    #     for security in changes.AddedSecurities:
    #         security.SetLeverage(10)

