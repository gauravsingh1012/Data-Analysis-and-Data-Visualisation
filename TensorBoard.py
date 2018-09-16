import datetime
import tensorflow as tf


class TensorBoard(object):
    """
    TensorBoard is a visualization tool provided with TensorFlow.
    This class can be used to record attributes from a running
    Zipline algorithm.
    """

    def __init__(self,arg):
        self.tbc = arg


    def log_dict(self, epoch, logs):
        """
        Writes a dictionary of simple named values to TensorBoard.
        Args:
            epoch: An integer representing time.
            logs: A dict containing what we want to log to TensorBoard.
        """
        for name, value in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.tbc.writer.add_summary(summary, global_step=epoch)
        self.writer.flush()

    def log_algo(self, algo, epoch=None, other_logs={}):
        """
        Logs info about a Zipline algorithm as it's running.
        Args:
            epoch: An integer representing algorithm time.
                   If None, then the algorithm's current
                   date is converted to an ordinal so that
                   these integers are monotonically increasing
                   with time. The same integer convention should
                   be used across different runs so that their charts
                   line up correctly.
           algo: An instance of a zipline.algorithm.TradingAlgorithm
           other_logs: A dictionary containing other things we want to log.
        """
        if epoch is None:
            epoch = datetime.date.toordinal(algo.get_datetime())

        logs = {}

        # add portfolio related things
        logs['portfolio value'] = algo.portfolio.portfolio_value
        logs['portfolio pnl'] = algo.portfolio.pnl
        logs['portfolio return'] = algo.portfolio.returns
        logs['portfolio cash'] = algo.portfolio.cash
        logs['portfolio capital used'] = algo.portfolio.capital_used
        logs['portfolio positions exposure'] = algo.portfolio.positions_exposure
        logs['portfolio positions value'] = algo.portfolio.positions_value
        logs['number of orders'] = len(algo.blotter.orders)
        logs['number of open orders'] = len(algo.blotter.open_orders)
        logs['number of open positions'] = len(algo.portfolio.positions)

        # add recorded variables from `zipline.algorithm.record` method
        for name, value in algo.recorded_vars.items():
            logs[name] = value

        # add any extras passed in through `other_logs` dictionary
        for name, value in other_logs.items():
            logs[name] = value

        self.log_dict(epoch, logs)
