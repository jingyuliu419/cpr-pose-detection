# File: build_trt_engine_rtmpose.py
import pycuda.autoinit  # 自动创建并管理上下文
import tensorrt as trt
import os

def build_engine(onnx_path: str, engine_path: str):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)

    with open(onnx_path, 'rb') as model:
        print(f"[INFO] Parsing ONNX file: {onnx_path}")
        if not parser.parse(model.read()):
            print("[ERROR] Failed to parse the ONNX file")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

    print("[INFO] Network inputs:")
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"  Input {i}: {input_tensor.name} {input_tensor.shape} {input_tensor.dtype}")

    print("[INFO] Network outputs:")
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"  Output {i}: {output_tensor.name} {output_tensor.shape} {output_tensor.dtype}")

    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, (1, 3, 256, 192), (1, 3, 256, 192), (1, 3, 256, 192))
    config.add_optimization_profile(profile)

    print("[INFO] Building TensorRT engine...")
    engine = builder.build_engine(network, config)
    if engine is None:
        print("[ERROR] Failed to build the engine.")
        return

    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"[SUCCESS] Engine saved at {engine_path}")

if __name__ == '__main__':
    ONNX_PATH = '/home/ljy/project/poseDetection/models/rtmpose_model/rtmpose.onnx'
    ENGINE_PATH = '/home/ljy/project/poseDetection/models/rtmpose_model/rtmpose.engine'
    build_engine(ONNX_PATH, ENGINE_PATH)
