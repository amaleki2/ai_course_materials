import os
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

    def _get(self, hw, q, k, default=None):
        if hw not in self.test_key:
            raise ValueError(f"Homework {hw} was not found in the test key.")

        if q not in self.test_key[hw]:
            raise ValueError(f"Question {q} was not found in the homework {hw} test key .")

        output = self.test_key[hw][q]
        if output:
            output = [o.get(k, default) for o in self.test_key[hw][q]]

        if not output:
            raise ValueError(f"{hw}_{q} did not have a value for {k}")

        return output

    def get_assert_type(self, hw, q):
        return self._get(hw, q, "assert_type", default="assert_equal")

    def get_args(self, hw, q):
        def transform(_args, func_list):
            if not func_list:
                return _args

            return [x if func is None else eval(func)(x) for (x, func) in zip(_args, func_list)]

        args_list = self._get(hw, q, "args", default=[])
        args_transformations_list = self.get_args_transformations(hw, q)
        args = [transform(args, args_transformations)
                for (args, args_transformations) in zip(args_list, args_transformations_list)]

        return args

    def get_args_transformations(self, hw, q):
        return self._get(hw, q, "args_transformations", default=[])

    def get_kwargs(self, hw, q):
        return self._get(hw, q, "kwargs", default={})

    def get_expected_output(self, hw, q):
        out = self._get(hw, q, "expected_output")
        out = [eval(o.replace('%eval ', '')) if isinstance(o, str) and o.startswith('%eval') else o for o in out]
        return out


class Tester:
    def __init__(self):
        current_folder = os.path.split(__file__)[0]
        if os.path.isfile("ai_course_materials/ai_course/test_keys.json"):  # python local relative path
            path = "ai_course_materials/ai_course/test_keys.json"
        elif os.path.isfile(os.path.join(current_folder, 'test_keys.json')):  # python local abs path
            path = os.path.join(current_folder, 'test_keys.json')
        elif os.path.isfile(os.path.join('/content', 'test_keys.json')):  # colab
            path = os.path.join('/content', 'test_keys.json')
        else:
            raise FileNotFoundError("test keys not found.")

        self.test_keys = TestExampleKeys(path)

    def assert_equal(self, a, b):
        if isinstance(a, str):
            return a == b
        if isinstance(a, dict):
            return a == b
        if not hasattr(a, '__len__'):  # is a scaler
            return a == b
        if isinstance(a, np.ndarray):
            return np.all(a == b)
        if isinstance(a, list) or isinstance(a, tuple):
            for (ai, bi) in zip(a, b):
                if not self.assert_equal(ai, bi):
                    return False
            return True

        return False

    def assert_almost_equal(self, a, b):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return ((a-b)**2).mean() < 1e-8
        elif isinstance(a, tuple) and isinstance(b, tuple):
            return self.assert_almost_equal(np.array(a), np.array(b))
        elif isinstance(a, list) and isinstance(b, list):
            return self.assert_almost_equal(np.array(a), np.array(b))
        elif isinstance(a, float) and isinstance(b, float):
            return abs(a-b) < 1e-6
        else:
            return False

    @staticmethod
    def _get_function_name(func):
        name = inspect.getsource(func)
        func_name = name.replace('def', '').replace(' ', '').split('(')[0]
        if func_name == '':
            raise(FunctionNameNotFoundException())
        return func_name

    def test_func(self, func):
        func_name = self._get_function_name(func)
        hw, question = func_name.split("_")[:2]

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

            if func_output is None:
                print("The function did not return any value")

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