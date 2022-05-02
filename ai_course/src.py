import os
import re
import inspect
import json

import numpy as np


class CustomException(Exception):
    def __init__(self, msg, error_msg=None):
        if error_msg is not None:
            msg += error_msg

        super().__init__(msg)


class FunctionNameNotFoundException(CustomException):
    def __init__(self, error_msg=None):
        msg = "_" * 50
        msg += "\nFunction name was not found. Did you change the function name?\n"
        super().__init__(msg, error_msg=error_msg)


class FunctionDoesNotRunException(CustomException):
    def __init__(self, error_msg=None):
        msg = "\n" + "_" * 75
        msg += "\nTHE FUNCTION RETURNED AN ERROR. See the message below to debug your code\n"
        msg += "_" * 75 + "\n"
        super().__init__(msg, error_msg=error_msg)


class TestFailedException(CustomException):
    def __init__(self, error_msg=None):
        msg = "\n" + "="*50 + "\n"
        msg += "\nTHE TEST FAILED.\n"
        msg += "  _________\n /  _   _  \\  \n |         |\n |    O    |\n \\_________/\n"
        super().__init__(msg, error_msg=error_msg)


class TestExampleKeys:
    def __init__(self, test_key_path):
        with open(test_key_path, 'r') as fid:
            self.test_key = json.load(fid)

    def _get(self, hw, q, k, must_return=False):
        if hw not in self.test_key:
            raise ValueError(f"Homework {hw} was not found in the test key.")

        if q not in self.test_key[hw]:
            raise ValueError(f"Question {q} was not found in the homework {hw} test key .")

        output = self.test_key[hw][q]
        if not output:
            output = None
        else:
            output = [o.get(k) for o in self.test_key[hw][q]]

        if must_return and (not output):
            raise ValueError(f"Test {hw}_{q} did not return a value for {k}.")
        return output

    def get_assert_type(self, hw, q):
        return self._get(hw, q, "assert_type")

    def get_args(self, hw, q):
        return self._get(hw, q, "args")

    def get_kwargs(self, hw, q):
        return self._get(hw, q, "kwargs")

    def get_expected_output(self, hw, q):
        return self._get(hw, q, "expected_output")


class Tester:
    current_folder = os.path.split(__file__)[0]
    TEST_KEY_PATH = os.path.join(current_folder, 'test_keys.json')

    def __init__(self):
        self.test_keys = TestExampleKeys(self.TEST_KEY_PATH)

    @staticmethod
    def assert_equal(a, b):
        return a == b

    @staticmethod
    def assert_almost_equal(a, b):
        return ((a-b)**2).mean() < 1e-8

    @staticmethod
    def assert_true(a):
        return a

    @staticmethod
    def _get_function_name(func):
        name = inspect.getsource(func)
        func_name = re.findall('def.*hw[0123456789]_q[0123456789]', name)[0].replace('def', '').strip()
        if func_name == '':
            raise(FunctionNameNotFoundException())
        return func_name

    def test_func(self, func):
        func_name = self._get_function_name(func)
        hw, question = func_name.split("_")

        args_list = self.test_keys.get_args(hw, question)
        kwargs_list = self.test_keys.get_kwargs(hw, question)
        assert_types = self.test_keys.get_assert_type(hw, question)
        expected_outputs = self.test_keys.get_expected_output(hw, question)
        for args, kwargs, assert_type, expected_output \
                in zip(args_list, kwargs_list, assert_types, expected_outputs):
            assert_func = getattr(self, assert_type)

            try:
                func_output = func(*args, **kwargs)
            except Exception as msg:
                error_msg = " ".join(msg.args)
                raise(FunctionDoesNotRunException(error_msg=error_msg))

            if assert_func(func_output, expected_output) is False:
                input_str = f"{args[0]}"
                for arg in args[1:]:
                    input_str = f"{input_str}, {arg}"
                for k, v in kwargs.items():
                    input_str = f"{input_str}, {k}={v}"
                error_msg = f"Input: {input_str}\nExpected Output: {expected_output} \nOutput : {func_output}"
                raise(TestFailedException(error_msg=error_msg))

            # unittest.main()
        print("="*25)
        print("No Error. Yaaaaay!!!")
        print('  _________\n /         \\\n |  /\\ /\\  |\n |    -    |\n |  \\___/  |\n \\_________/')
        print("="*25)