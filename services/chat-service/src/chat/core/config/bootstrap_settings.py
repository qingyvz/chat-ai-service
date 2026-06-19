from common.core.config.bootstrap_settings import BootstrapSettings


class ChatBootstrapSettings(BootstrapSettings):
    """
    chat-service 引导配置
    """

    APP_NAME: str = "Chat Service"
    SERVICE_NAME: str = "chat-service"
    SERVICE_PORT: int = 19904

bootstrap_settings = ChatBootstrapSettings()
