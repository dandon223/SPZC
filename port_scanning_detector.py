import pickle
import numpy as np
import pandas as pd
import pyshark
import struct



# def in_thread(packet, bayes):
#	data = retrieve_attributes(packet)
#	bayes.fit(data, [0])
#	Predict_X = bayes.predict(data)
#	print('Predict_X', Predict_X)

def retrieve_attributes(packet):
    pkt_to_list = []

    # List of the attributes to be retrieved from each packet (wireshark.org/docs/dfref/)

    attributes = [
        ["frame_info", "encap_type"],  #
        ["frame_info", "time"],  #
        ["frame_info", "time_epoch"],  #
        ["frame_info", "number"],  #
        ["frame_info", "len"],  #
        ["frame_info", "cap_len"],  #
        ["eth", "type"],  # Ethernet Type
        ["ip", "version"],  # Internet Protocol (IP) Version
        ["ip", "hdr_len"],  # IP header length (IHL)
        ["ip", "tos"],  # IP Type of Service (TOS)
        ["ip", "id"],  # Identification
        ["ip", "flags"],  # IP flags
        ["ip", "flags.rb"],  # Reserved bit flag
        ["ip", "flags.df"],  # Don't fragment flag
        ["ip", "flags.mf"],  # More fragments flag
        ["ip", "frag_offset"],  # Fragment offset
        ["ip", "ttl"],  # Time to live
        ["ip", "proto"],  # Protocol (e.g. tcp == 6)
        ["ip", "checksum"],  # Header checksum (qualitative)
        ["ip", "src"],  # Source IP Address
        ["ip", "dst"],  # Destination IP Address
        ["ip", "len"],  # Total length
        ["ip", "dsfield"],  # Differentiated Services Field

        ["tcp", "srcport"],  # TCP source port
        ["tcp", "dstport"],  # TCP Destination port
        ["tcp", "seq"],  # Sequence number
        ["tcp", "ack"],  # Acknowledgment number
        ["tcp", "len"],  # TCP segment length
        ["tcp", "hdr_len"],  # Header length
        ["tcp", "flags"],  # Flags
        ["tcp", "flags.fin"],  # FIN flag
        ["tcp", "flags.syn"],  # SYN flag
        ["tcp", "flags.reset"],  # RST flag
        ["tcp", "flags.push"],  # PUSH flag
        ["tcp", "flags.ack"],  # ACK flag
        ["tcp", "flags.urg"],  # URG flag
        ["tcp", "flags.cwr"],  # Congestion Window Reduced (CWR) flags
        ["tcp", "window_size"],  # Window Size
        ["tcp", "checksum"],  # Checksum
        ["tcp", "urgent_pointer"],  # Urgent pointer
        ["tcp", "options.mss_val"]  # Maximum Segment Size
    ]

    columns = []
    for i in attributes:
        columns.append(str(i[0]) + "." + str(i[1]))

    for i in attributes:
        # try-except used for packet attribute validation, if not available, fill with ""
        try:
            attribute = getattr(packet, i[0])
            pkt_to_list.append(getattr(attribute, i[1]))
        except:
            pkt_to_list.append("")
    df = pd.DataFrame(np.empty((0, 41)))
    df.columns = columns
    df.loc[0] = pkt_to_list
    print(df['ip.src'].values + ' ' + df['ip.dst'].values)
    return df


filename = 'model/dt_model.sav'
bayes = pickle.load(open(filename, 'rb'))

capture = pyshark.LiveCapture(interface='Wi-Fi', display_filter="tcp")
for packet in capture.sniff_continuously():
    data = retrieve_attributes(packet)

    data.drop(columns=['frame_info.time', 'frame_info.encap_type', 'frame_info.time_epoch', 'frame_info.number',
                       'frame_info.len', 'frame_info.cap_len', 'eth.type', 'ip.flags', 'ip.src', 'ip.dst',
                       'ip.version', 'ip.proto', 'tcp.flags'], axis=1, inplace=True)
    data.drop(columns=["ip.checksum", "ip.ttl", "tcp.checksum", "tcp.dstport", "tcp.seq", "tcp.srcport",
                       "tcp.ack", "tcp.options.mss_val", 'ip.hdr_len', 'ip.tos', 'ip.flags.rb',
                       'ip.flags.mf', 'ip.frag_offset'], axis=1, inplace=True)

    data = data.replace('', np.nan, regex=True)
    data = data.fillna(0)
    fields = ['ip.id', 'ip.dsfield']

    for field in fields:
        data[field] = data[field].apply(lambda x: int(str(x), 16))
    # poniewaz obie zbiory maja rozne parametry, trzeba zdecydowac co bierzemy pod uwage
    Predict_X = bayes.predict(data)
    print('Predict_X', Predict_X)


