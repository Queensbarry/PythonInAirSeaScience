from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
import logging

import statsmodels.api as sm
import os

import numpy as np
import pandas as pd
from itertools import repeat
from multiprocessing import Pool
import math

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit
import statsmodels.api as sm



class _Tool():
    def KS(y_true, y_hat, sample_weight=None):
        if isinstance(y_true, np.ndarray):
            y_true = pd.Series(y_true)
        if sample_weight is None:
            sample_weight = pd.Series(np.ones_like(y_true), index=y_true.index)
        if isinstance(y_hat, np.ndarray):
            y_hat = pd.Series(y_hat, index=y_true.index)
        sample_weight.name = 'sample_weight'
        y_true.name = 'y'
        y_hat.name = 'score'
        df = pd.concat([y_hat, y_true, sample_weight], axis=1)
        df['y_mutli_w'] = df['y'] * df['sample_weight']
        total = df.groupby(['score'])['sample_weight'].sum()
        bad = df.groupby(['score'])['y_mutli_w'].sum()
        all_df = pd.DataFrame({'total': total, 'bad': bad})
        all_df['good'] = all_df['total'] - all_df['bad']
        all_df.reset_index(inplace=True)
        all_df = all_df.sort_values(by='score', ascending=False)
        all_df['badCumRate'] = all_df['bad'].cumsum() / all_df['bad'].sum()
        all_df['goodCumRate'] = all_df['good'].cumsum() / all_df['good'].sum()
        ks = all_df.apply(lambda x: x.goodCumRate - x.badCumRate, axis=1)
        return np.abs(ks).max()

    def vif(df):
        vif = pd.DataFrame()
        vif['features'] = df.columns
        if df.shape[1] > 1:
            vif['VIF Factor'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        else:
            vif['VIF Factor'] = 0
        vif = vif.sort_values('VIF Factor', ascending=False)
        return vif

    def make_logger(logger_name, logger_file):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logger_file, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(name)s]-[%(filename)s-%(lineno)d]-[%(processName)s]-[%(asctime)s]-[%(levelname)s]: %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger, fh


SCORERS = dict(
    r2=r2_score,
    explained_variance_score=explained_variance_score,
    max_error=max_error,
    accuracy=accuracy_score,
    roc_auc=roc_auc_score,
    balanced_accuracy=balanced_accuracy_score,
    average_precision=average_precision_score,
    ks=_Tool.KS)


class Regression():

    def __init__(self, X, y, fit_weight=None, measure='ks', measure_weight=None, kw_measure_args=None,
                 max_pvalue_limit=0.05, max_vif_limit=3, max_corr_limit=0.6, coef_sign=None, iter_num=20,
                 kw_algorithm_class_args=None, n_core=None, logger_file_CH=None, logger_file_EN=None):
        self.X = X
        self.y = y
        self.fit_weight = fit_weight
        self.measure = measure
        self.kw_measure_args = {'sample_weight': measure_weight}
        if kw_measure_args is not None:
            self.kw_measure_args.update(kw_measure_args)
        self.max_pvalue_limit = max_pvalue_limit
        self.max_vif_limit = max_vif_limit
        self.max_corr_limit = max_corr_limit
        self.coef_sign = coef_sign
        self.iter_num = iter_num
        self.kw_algorithm_class_args = kw_algorithm_class_args
        self.logger_file_CH = logger_file_CH
        self.logger_file_EN = logger_file_EN
        if n_core is None:
            self.n_core = os.cpu_count() - 1
        elif n_core >= 1:
            self.n_core = n_core
        else:
            self.n_core = math.ceil(os.cpu_count() * n_core)

    def _check(self, clf, in_vars, current_perf):
        X = self.X[in_vars]
        check_param = True
        if isinstance(self.coef_sign, dict):
            coef_pos = {k: v for k, v in self.coef_sign.items() if v == '+'}
            if len(coef_pos) > 0:
                check_param = (clf.params[clf.params.index.isin(coef_pos)] > 0).all()
                if check_param:
                    coef_neg = {k: v for k, v in self.coef_sign.items() if v == '-'}
                    if len(coef_neg) > 0:
                        check_param = (clf.params[clf.params.index.isin(coef_neg)] < 0).all()
        elif self.coef_sign == '+':
            check_param = (clf.params[1:] > 0).all()
        elif self.coef_sign == '-':
            check_param = (clf.params[1:] < 0).all()

        check_pvalue = (clf.pvalues < self.max_pvalue_limit).all()
        y_hat = pd.Series(clf.predict(sm.add_constant(X)), index=self.y.index, name='score')
        perf = SCORERS[self.measure](self.y, y_hat, **self.kw_measure_args)
        check_perf = perf > current_perf
        if X.shape[1] < 2:
            corr_max = 0
            vif = 0
        else:
            vif = _Tool.vif(X).iloc[0, 1]
            df_corr = X.corr()
            t = np.arange(df_corr.shape[1])
            df_corr.values[t, t] = np.nan
            corr_max = df_corr.max().max()
        check_vif = vif < self.max_vif_limit
        check_corr = corr_max < self.max_corr_limit

        return check_param, check_pvalue, check_perf, perf, check_vif, vif, check_corr, corr_max

    def _add_var(self, args):
        col = args[0]
        in_vars, current_perf = args[1]
        add_rm_var = (None, None, current_perf)
        tmp_cols = [col]
        tmp_cols.extend(in_vars)
        clf = self._regression(tmp_cols)
        check_param, check_pvalue, check_perf, perf, check_vif, vif, check_corr, corr_max = self._check(clf, tmp_cols,
                                                                                                        current_perf)
        if check_perf:
            if check_param and check_pvalue and check_vif and check_corr:
                add_rm_var = (col, None, perf)
            else:
                if len(in_vars) > 0:
                    rm_var_arr = map(self._rm_var, zip(in_vars, repeat((tmp_cols, current_perf))))
                    rm_var, perf = sorted(rm_var_arr, key=lambda x: x[1])[-1]
                    if rm_var:
                        add_rm_var = (col, rm_var, perf)
        return add_rm_var

    def _rm_var(self, args):
        col = args[0]
        in_vars, current_perf = args[1]
        rm_var = (None, current_perf)
        X_tmp = self.X[in_vars]
        X_tmp = X_tmp.loc[:, X_tmp.columns != col]
        clf = self._regression(list(X_tmp.columns))
        check_param, check_pvalue, check_perf, perf, check_vif, vif, check_corr, corr_max = self._check(clf, list(
            X_tmp.columns), current_perf)
        check_pass = (check_param and check_pvalue and check_vif and check_corr and check_perf)
        if check_pass:
            rm_var = (col, perf)
        return rm_var

    def _del_reason(self, args):
        col = args[0]
        in_vars, current_perf = args[1]
        tmp_cols = [col]
        tmp_cols.extend(in_vars)

        clf = self._regression(tmp_cols)
        check_param, check_pvalue, check_perf, perf, check_vif, vif, check_corr, corr_max = self._check(clf, tmp_cols,
                                                                                                        current_perf)

        reasons = []
        reasons_en = []
        if not check_perf:
            reasons.append('模型性能=%f,小于等于最终模型的性能=%f' % (perf, current_perf))
            reasons_en.append(
                'the performance index of model=%f,less or equals than the performance index of final model=%f' % (
                perf, current_perf))
        if not check_vif:
            reasons.append('最大VIF=%f,大于设置的阈值=%f' % (vif, self.max_vif_limit))
            reasons_en.append('the max VIF=%f,more than the setting of max_vif_limit=%f' % (vif, self.max_vif_limit))
        if not check_corr:
            reasons.append('最大相关系数=%f,大于设置的阈值=%f' % (corr_max, self.max_corr_limit))
            reasons_en.append('the max correlation coefficient=%f,more than the setting of max_corr_limit=%f' % (
            corr_max, self.max_corr_limit))
        if not check_pvalue:
            reasons.append('有些系数不显著，P_VALUE大于设置的阈值=%f' % (self.max_pvalue_limit))
            reasons_en.append(
                'some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=%f' % (
                    self.max_pvalue_limit))
        if not check_param:
            reasons.append('有些系数不符合coef_sign的设置')
            reasons_en.append('some setting of coef_sign are unreachable')
        return (col, reasons, reasons_en)

    def fit(self):
        '''
        中文版
        训练模型


        Returns
        -------
        in_vars : list
        所有的入模变量列表，列表中的顺序即为加入时的顺序

        clf_final : statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        最终的逐步回归的模型

        dr : dict
        删除原因，其结构为:{'变量名称':([...],[...])}
        每个dr的value是一个含有两个元素的tuple，第一个为中文给出的删除原因，第二个为英文给出的删除原因。每个元素是一个list，记录了对应变量(key)的所有删除原因。如果某一个特征对应的list里没有任何元素，则应考虑将这个特征手工的加入到模型中去。

        English Document
        Fitting a model


        Returns
        -------
        in_vars : list
        All variables to be picked up by model.The order in list is same with the order of to be added

        clf_final : statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        A final step-wise model

        dr : dict
        deletion reason.It`s format is {'var_name':([...],[...])}
        Every value in dr contains a tuple including two elements.The first element is reason in Chinese and the second in English.Every element is a list and record all deletion reason of variable(matching key).Some features should be added into model manually,if a list corresponding these features has no any element.
        '''
        c = 0
        in_vars = []
        current_perf = -np.inf
        logger_ch = None
        logger_en = None
        if self.logger_file_CH is not None:
            logger_ch, fh_ch = _Tool.make_logger('LogisticReg_Step_Wise_MP_CH', self.logger_file_CH)

        if self.logger_file_EN is not None:
            logger_en, fh_en = _Tool.make_logger('LogisticReg_Step_Wise_MP_EN', self.logger_file_EN)

        while True:
            c += 1
            if c > self.iter_num:
                break
            if logger_ch:
                logger_ch.info('****************迭代轮数：%d********************' % c)
            if logger_en:
                logger_en.info('****************Iterate Number:%d********************' % c)

            out_vars = self.X.columns[~self.X.columns.isin(in_vars)]
            if len(out_vars) == 0:
                if logger_ch:
                    logger_ch.info('变量全部进入模型，建模结束')
                if logger_en:
                    logger_en.info('All variables are picked by step model. Modeling is completed!')
                break
            with Pool(self.n_core) as pool:
                result = pool.map_async(self._add_var, zip(out_vars, repeat((in_vars, current_perf))))
                add_rm_var_arr = result.get()
            add_var, rm_var_0, perf = sorted(add_rm_var_arr, key=lambda x: x[2])[-1]
            if add_var is not None:
                in_vars.append(add_var)
                current_perf = perf
                if rm_var_0:
                    in_vars.remove(rm_var_0)
            if len(in_vars) == 0:
                if logger_ch:
                    logger_ch.info('没有变量能够进入模型，建模结束')
                if logger_en:
                    logger_en.info('All variables can`t be picked by step model. Modeling is completed!')
                break
            with Pool(self.n_core) as pool:
                result = pool.map_async(self._rm_var, zip(in_vars, repeat((in_vars, current_perf))))
                rm_var_arr = result.get()
            rm_var, perf = sorted(rm_var_arr, key=lambda x: x[1])[-1]
            if rm_var is not None:
                in_vars.remove(rm_var)
                current_perf = perf
            if (add_var is None) and (rm_var is None):
                if logger_ch:
                    logger_ch.info('在此轮迭代中，在满足使用者所设置条件的前提下，已经不能通过增加或删除变量来进一步提升模型的指标，建模结束')
                if logger_en:
                    logger_en.info(
                        'At this iteration,it`s not reachable under conditions you set that promoting performance index of model by adding or removing any variable. Modeling is completed!')
                break
            if logger_ch:
                logger_ch.info('此轮迭代完成，当前入模变量为：%s。 当前模型性能%s为:%f' % (in_vars, self.measure, current_perf))
            if logger_en:
                logger_en.info(
                    'This iteration is end.Current variables in model are %s.The performance of model is %s=%f' % (
                    in_vars, self.measure, current_perf))

        clf_final = self._regression(in_vars)
        out_vars = self.X.columns[~self.X.columns.isin(in_vars)]
        with Pool(self.n_core) as pool:
            result = pool.map_async(self._del_reason, zip(out_vars, repeat((in_vars, current_perf))))
            del_var_arr = result.get()
        dr = dict((col, (reasons, reasons_en)) for col, reasons, reasons_en in del_var_arr)
        if logger_ch:
            fh_ch.close()
            logger_ch.removeHandler(fh_ch)
        if logger_en:
            fh_en.close()
            logger_en.removeHandler(fh_en)
        return in_vars, clf_final, dr


class LinearReg(Regression):
    '''
    中文版文档(Document in English is in the next.）
    MultiProcessMStepRegression.LinearReg:多进程逐步线性回归，其底层的线性回归算法使用的是statsmodels.api.OLS或statsmodels.api.WLS，依据用户是否使用训练样本权重来绝定。
    每一次向前添加过程中都会使用多进程来同时遍历多个解释变量，然后选取其中符合使用者设定的条件且能给线性回归带来最大性能提升的解释变量加入到模型中，如果所有变量都不能在满足使用者设置条件的前提下提升模型性能，则此次添加过程不加入任何变量。
    每一次的向后删除过程中也使用与向前添加过程同样的原则来决定删除哪个变量。
    在添加过程中模型性能有提升，但是部分条件不被满足，此时会额外触发一轮向后删除的过程，如果删除的变量与正要添加的变量为同一个，则此变量不被加入，添加流程结束。如果删除的变量与正要添加的变量不是同一个，则添加当前的变量，并将需要删除的变量从当前选中变量列表中排除。额外触发的向后删除过程与正常的向后删除过程的流程一致。
    在建模结束后，会将没有入选的解释变量分别加入到现有模型变量中，通过重新建模，会给出一个准确的没有入选该变量的原因。
    支持的功能点如下：
    1.支持双向逐步回归(Step_Wise)
    2.支持多进程，在每步增加变量或删除变量时，使用多进程来遍历每个候选变量。Windows系统也支持多进程。
    3.支持使用者指定的指标来作为变量添加或删除的依据，而不是使用AIC或BIC，在处理不平衡数据时可以让使用者选择衡量不平衡数据的指标
    4.支持使用者指定P-VALUE的阈值，如果超过该阈值，即使指标有提升，也不会被加入到变量中
    5.支持使用者指定VIF的阈值，如果超过该阈值，即使指标有提升，也不会被加入到变量中
    6.支持使用者指定相关系数的阈值，如果超过该阈值，即使指标有提升，也不会被加入到变量中
    7.支持使用者指定回归系数的正负号，在某些业务中，有些特征有明显的业务含义，例如WOE转换后的数据，就会要求回归系数均为正或均为负，加入对系数正负号的限制，如果回归系数不满足符号要求，则当前变量不会被加入到变量中
    8.上述4，5，6，7均在逐步回归中完成，挑选变量的同时校验各类阈值与符号
    9.会给出每一个没有入模变量被剔除的原因，如加入后指标下降，P-VALUE超出指定阈值，正负号与使用者的预期不符等等。
    10.支持中英文双语的日志，会将逐步回归中的每一轮迭代的情况记录到中文日志和英文日志中

    注意：因为该类会将数据X和y作为该类一个实例的属性，所以实例会比较大，因此非必要时，尽量不要保存MultiProcessMStepRegression.LinearReg的实例。而是保存其返回的模型和删除原因等信息。


    Parameters
    ----------
    X:DataFrame
    features

    y:Series
    target

    fit_weight:Series
    长度与样本量相同，为训练模型时的weight，如果取值为None（默认），则认为各个样本的训练权重相同，选用statsmodels.api.OLS做为底层的实现算法。如果不为空，则会选用statsmodels.api.WLS做为底层的实现算法。在线性回归中设置权重的目的是，在异方差的情况下，训练出稳定的模型。

    measure:str r2(默认) | explained_variance_score | max_error
    计算线性回归模型性能的函数，y_true,y_hat和measure_weight会被自动传递进指定measure函数中，其余参数会由kw_measure_args传入

    measure_weight:Series
    长度与样本量相同，为度量模型性能时的weight，如果取值为None（默认），则认为各个样本的度量权重相同。

    kw_measure_args:dict | None(默认)
    measure函数除y_true,y_hat,measure_weight外，其余需要传入的参数都写入该dict里。None意味着不需要传入额外的参数

    max_pvalue_limit:float
    允许的P-VALUE的最大值。0.05（默认）
    max_vif_limit:float
    允许的VIF的最大值。3（默认）

    max_corr_limit:float
    允许的相关系数的最大值。0.6（默认）

    coef_sign:'+','-',dict,None（默认）
        如果知道X对y的影响关系--正相关或负相关，则可以对变量的符号进行约束。
        '+':所有X的系数都应为正数
        '-':所有X的系数都应为负数
        dict:格式如{'x_name1':'+','x_name2':'-'}，将已知的X的系数符号配置在dict中，以对回归结果中X的系数的正负号进行约束。没有被包含在dict中的变量，不对其系数进行约束
        None:所有X的系数的正负号都不被约束

    iter_num:int
    挑选变量的轮数，默认为20。np.inf表示不限制轮数，当变量很多时，需要较长的运行时间。如果所有的变量都已经被选入到模型，或者不能通过增加或删除变量来进一步提升模型性能，则实际迭代轮数可能小于iter_num。每一轮挑选变量包含如下步骤：1.尝试将每一个还未被加入到模型中的变量加入到当前模型中，选出一个满足使用者设置的条件且使模型性能提升最多的变量加入到模型中。2.在当前模型中的每一个变量尝试删除，选出一个满足使用者设置的条件且使模型性能提升最多的变量移出模型。完成1，2两步即为完成一轮迭代。如果步骤1和2均未能挑选出变量，则迭代提前终止，无论是否达到了iter_num。

    kw_algorithm_class_args:dict
    除X，y，fit_weight外，其它需要传入线性回归算法（OLS，WLS）的参数。


    n_core:int | float | None
    CPU的进程数。如果是int类型，则为使用CPU的进程数。如果是float类型，则为CPU全部进程数的百分比所对应的进程数（向上取整）。如果为None，则为使用全部CPU进程数-1

    logger_file_CH:str
    使用者指定的用于记录逐步回归过程的文件名，日志为中文日志。如果为None（默认）则不记录中文日志

    logger_file_EN:str
    使用者指定的用于记录逐步回归过程的文件名，日志为英文日志。如果为None（默认）则不记录英文日志
    Document in English
    MultiProcessMStepRegression.LinearReg:A Step-Wise Linear Regression handling with multi-processing.It bases on statsmodels.api.OLS or statsmodels.api.WLS supplying a linear regression algorithm.Which algorithm should be used depends on the setting of train sample weight.
    In adding feature process,multi-processing is used to traversal several features concurrently.The feature which meets the conditions which the user set and get a max lift on measure index is added in the model.If any feature can`t improve the performance of model undering the conditions set by user ,no feature is added in current iteration.
    The removing feature process has same policy with adding feature process to decide which feature should be removed.
    When adding process, if there is improving on performance of model but some conditions user set are missed,a additional removing process will start to run.If the feature to remove is same with the feature to add,the feature will not be added and the adding process is over.If They are not same,the feature to add is added in and the feature to remove is excluded from current list in which the picked features stay.The additional removing process has same procedure with removing process.
    When modeling is compeleted,the features not picked up will respectively be added in picked features list. And then by rebuilding model with those features,a exact deletion reasons will return.

    The characteristics are listed below:
    1.Supporting forward-backward Step-Wise.
    2.Supporting multi-processing.When adding or removing features,multi-processing is used to traversal all candidate features.
    3.Supporting that user could point the index instead of AIC/BIC for measuring model performance when adding or removing feaures.That is benifit when user`s data is unbalanced.
    4.Supporting that user could point p-value threshold.If max p-value is more than this threshold,the current features will not be added,although getting a lift on performance of model.
    5.Supporting that user could point VIF threshold.If max VIF is more than this threshold,the current features will not be added,although getting a lift on performance of model.
    6.Supporting that user could point coefficient of correlation threshold.If max coefficient of correlation is more than this threshold,the current features will not be added,although getting a lift on performance of model.
    7.Supporting that user could point sign to coefficients of regression. A part of features have sense in some business like woe transfer which require that all coefficients of regression are postive or negtive.If the signs requirement is not met,the current features will not be added,although getting a lift on performance of model.
    8.[4,5,6,7] above are completed in step-wise procedure.Picking features and verifing those thresholds and signs are simultaneous.
    9.Users will get reasons of which features isn`t picked up,as performance is fall or p-value is more than threshold or signs is not in accord with user`s expect and so on after adding this feature
    10.Supporting the Chinese and English log in whcih user can get record of every iteration

    Note:As X and y is a property in a instance of MultiProcessMStepRegression.LinearReg class,so that instance will be very large.Saving that instance is not recommended instead of saving the returned model and remove reasons.


    Parameters
    ----------
    X:DataFrame
    features

    y:Series
    target

    fit_weight:Series
    The length of fit_weight is same with length of y.The fit_weight is for trainning data.If None(default),every sample has a same trainning weight and statsmodels.api.OLS is used as base linear algorithm.If not None,statsmodels.api.WLS is used as base linear algorithm.In linear regression,the goal of setting weight is for getting a stable model with Heteroscedasticity.


    measure:str r2(默认) | explained_variance_score | max_error
    Performance evaluate function.The y_true,y_hat and measure_weight will be put into measure function automatically and the other parameters will be put into measure function with kw_measure_args

    measure_weight:Series
    The length of measure_weight is same with length of y.The measure_weight is for measuring function.If None(default),every sample has a same measuring weight.
    See also fit_weight

    kw_measure_args:dict | None(默认)
    Except y_true,y_hat and measure_weight,the other parameters need be put in kw_measure_args to deliver into measure function.
    None means that no other parameters delivers into measure function.

    max_pvalue_limit:float
    The max P-VALUE limit.
    0.05(default)
    max_vif_limit:float
    The max VIF limit.
    3(default)

    max_corr_limit:float
    The max coefficient of correlation limit.
    0.6(default)

    coef_sign:'+','-',dict,None（默认）
    If the user have a priori knowledge on relation between X and y,like positive correlation or negtive correlation,user can make a constraint restriction on sign of resression coefficient by this parameter.
    '+':all signs of resression coefficients are positive
    '-':all signs of resression coefficients are negtive
    dict:the format is as {'x_name1':'+','x_name2':'-'}.Put coefficient and coefficient`s sign on which you have a priori knowledge into a dict and then constraint these signs that are in this dict. The coefficients not included in this dict will not be constrainted.
    None:all coefficients are not constrainted.

    iter_num:int
    The iteration num for picking features.Default is 20.When np.inf,no limit to iteration num,if features are many,then the running time is long.If all features are already picked in model or no imporve on perfermance by adding/removing any feature,the actual iteration num should be samller than iter_num.The steps inclueed in every iteration is:1.Try adding feature which is not added in current model yet and then pick up one feature that makes most promotion for performance of model with satisfying user`s setting. 2.Try removing feature and then remove out one feature that makes most promotion for performance of model with satisfying user`s setting.It is means finshing one time iteration that step 1 and step 2 is completed.If all step 1 and step 2 can`t pick up any feature then iteration is pre-terminated,no matter whether iter_num is reached.

    kw_algorithm_class_args:dict
    Except X，y，fit_weight,the other parameters that are delivered into linear regression algorithm is in kw_algorithm_class_args
    Note:y,X is called endog and exog in statsmodels.genmod.generalized_linear_model.GLM

    n_core:int | float | None
    Count of CPU processing.If int,user point the count.If float,the count is as percentage of all count transfered to int(ceil).If None(default),all count of CPU processing -1.

    logger_file_CH:str
    A log file name where log for step-wise procedure is recorded in Chinese.If None(default),not recording Chinese log.

    logger_file_EN:str
    A log file name where log for step-wise procedure is recorded in English.If None(default),not recording English log.

    '''

    def __init__(self, X, y, fit_weight=None, measure='r2', measure_weight=None, kw_measure_args=None,
                 max_pvalue_limit=0.05, max_vif_limit=3, max_corr_limit=0.6, coef_sign=None, iter_num=20,
                 kw_algorithm_class_args=None, n_core=None, logger_file_CH=None, logger_file_EN=None):
        Regression.__init__(self, X, y, fit_weight, measure, measure_weight, kw_measure_args, max_pvalue_limit,
                            max_vif_limit, max_corr_limit, coef_sign, iter_num, kw_algorithm_class_args, n_core,
                            logger_file_CH, logger_file_EN)

    def _regression(self, in_vars):
        X = self.X[in_vars]
        if self.fit_weight is None:
            if self.kw_algorithm_class_args is not None:
                reg = sm.OLS(self.y, sm.add_constant(X), **self.kw_algorithm_class_args)
            else:
                reg = sm.OLS(self.y, sm.add_constant(X))
        else:
            if self.kw_algorithm_class_args is not None:
                reg = sm.WLS(self.y, sm.add_constant(X), weights=self.fit_weight, **self.kw_algorithm_class_args)
            else:
                reg = sm.WLS(self.y, sm.add_constant(X), weights=self.fit_weight)
        clf = reg.fit()
        clf.intercept_ = [clf.params.const]
        clf.coef_ = [clf.params[1:]]
        return clf
