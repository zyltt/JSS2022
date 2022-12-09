# coding=utf-8

from Method.asyncTournamentSelect import asyncTournamentSelect
from DataStruct.globalConfig import GlobalConfig
from Method.mutation import mutation
class Controller:
    a=0
    def __init__(self):
        return
    def excute(self):
        # TODO
        list=asyncTournamentSelect(GlobalConfig.P)
        for g in list:
            mutation(g)
            GlobalConfig.Q.push(g)
        return