import argparse
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


def main():
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board

    board = BoardShim(args.board_id, params)
    board.prepare_session()
    board.start_stream()
    print("Streaming data... Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
            # Find out how many packages have been streamed to the buffer
            data_count = board.get_board_data_count()
            
            # Optionally grab a snapshot of the newest data without removing it from the buffer
            # 'get_current_board_data(1)' grabs just the very last collected sample
            latest_data = board.get_current_board_data(1) 
            
            if latest_data.size > 0:
                # Assuming EEG channels start at row 1 for Cyton Daisy
                latest_eeg_val = latest_data[1][0] 
                print(f"Live Update: {data_count} total samples buffered. Latest Ch1 Value: {latest_eeg_val:.2f} µV")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Stopping stream...")
    finally:
        data = board.get_board_data()  # get all data and remove it from internal buffer
        board.stop_stream()
        board.release_session()
        print(f"Stream stopped. Collected {data.shape[1]} samples.")

    print(data)


if __name__ == "__main__":
    main()