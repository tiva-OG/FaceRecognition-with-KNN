def error_message_detail(error, error_detail) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occurred in python script; Name: [{0}]   Line no.: [{1}] Error message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
