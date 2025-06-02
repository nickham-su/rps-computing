# RPC服务管理器
from multiprocessing.managers import BaseManager

RPC_SERVER_ADDRESS = ('localhost', 50000)  # 默认RPC服务器地址、端口
RPC_SERVER_AUTHKEY: bytes = b'direct_router'  # 默认RPC服务器认证密钥


class RPCManager(BaseManager):
    pass


def get_direct_router():
    """获取DirectRouter"""
    from src.services.direct_router.direct_router import DirectRouter
    return DirectRouter


def get_multi_stop_router():
    from src.services.multi_stop_router.multi_stop_router import MultiStopRouter
    """获取MultiStopRouter"""
    return MultiStopRouter


# 注册RPC方法
RPCManager.register('get_direct_router', callable=get_direct_router)
RPCManager.register('get_multi_stop_router', callable=get_multi_stop_router)
