'''
Copied out of matrexapi/Agents/KryptowireAgents/Utility/utils.py
'''

from copy import deepcopy
from prettytable import PrettyTable
from CybORG.Shared.Actions import *
import numpy as np
from ipaddress import IPv4Network, IPv4Address


def map_action(numeric_action, action_mapping_dict):
    mapped_action = action_mapping_dict[int(numeric_action)]
    action = numeric_to_cyborg_action(mapped_action)
    return action


def numeric_to_cyborg_action(mapped_action):
    action = None
    hostname = mapped_action["hostname"]

    subnet = mapped_action["subnet"]
    if subnet:
        subnet = IPv4Network(subnet, strict=False)

    ip_address = mapped_action["ip_address"]
    if ip_address:
        ip_address = IPv4Address(mapped_action["ip_address"])
    target_session = mapped_action["target_session"]

    session = 0

    if mapped_action["name"] == "Sleep":
        action = Sleep()

    elif mapped_action["name"] == "Monitor":
        action = Monitor(session=session, agent='Blue')

    elif mapped_action["name"] == "Analyse":
        action = Analyse(hostname = hostname, session=session, agent='Blue')

    elif mapped_action["name"] == "Remove":
        action = Remove(hostname = hostname, session=session, agent='Blue')

    elif mapped_action["name"] == "DecoyApache":
        action = DecoyApache(hostname = hostname, session=session, agent='Blue')

    elif mapped_action["name"] == "DecoyFemitter":
        action = DecoyFemitter(hostname = hostname, session=session, agent='Blue')

    elif mapped_action["name"] == "DecoyHarakaSMPT":
                action = DecoyHarakaSMPT(hostname = hostname, session=session, agent='Blue')

    elif mapped_action["name"] == "DecoySmss":
        action = DecoySmss(hostname = hostname, session=session, agent='Blue')

    elif mapped_action["name"] == "DecoySSHD":
        action = DecoySSHD(hostname = hostname, session=session, agent='Blue')

    elif mapped_action["name"] == "DecoySvchost":
        action = DecoySvchost(hostname = hostname, session=session, agent='Blue')

    elif mapped_action["name"] == "DecoyTomcat":
        action = DecoyTomcat(hostname = hostname, session=session, agent='Blue')

    elif mapped_action["name"] == "DecoyVsftpd":
        action = DecoyVsftpd(hostname = hostname, session=session, agent='Blue')

    elif mapped_action["name"] == "Restore":
        action = Restore(hostname = hostname, session=session, agent='Blue')

    elif mapped_action["name"] == "DiscoverRemoteSystems":
        action = DiscoverRemoteSystems(session=session, agent = "Red", subnet=subnet)

    elif mapped_action["name"] == "DiscoverNetworkServices":
        action = DiscoverNetworkServices(session=session, agent = "Red", ip_address=ip_address)

    elif mapped_action["name"] == "ExploitRemoteService":
        if "priority" in mapped_action:
            priority = mapped_action["priority"]
            action = ExploitRemoteService(session=session, agent = "Red", ip_address=ip_address, priority=priority)
        else:
            action = ExploitRemoteService(session=session, agent = "Red", ip_address=ip_address)

    elif mapped_action["name"] == "PrivilegeEscalate":
        action = PrivilegeEscalate(session=session, agent = "Red", hostname=hostname)

    elif mapped_action["name"] == "Impact":
        action = Impact(session=session, agent = "Red", hostname=hostname)

    elif mapped_action["name"] == "BlueKeep":
        action = BlueKeep(session=session, agent = "Red", target_session = target_session, ip_address=ip_address)

    elif mapped_action["name"] == "EternalBlue":
        action = EternalBlue(session=session, agent = "Red", target_session = target_session, ip_address=ip_address)

    elif mapped_action["name"] == "FTPDirectoryTraversal":
                action = FTPDirectoryTraversal(session=session, agent = "Red", target_session = target_session, ip_address=ip_address)

    elif mapped_action["name"] == "HarakaRCE":
        action = HarakaRCE(session=session, agent = "Red", target_session = target_session, ip_address=ip_address)

    elif mapped_action["name"] == "HTTPRFI":
        action = HTTPRFI(session=session, agent = "Red", target_session = target_session, ip_address=ip_address)

    elif mapped_action["name"] == "HTTPSRFI":
        action = HTTPSRFI(session=session, agent = "Red", target_session = target_session, ip_address=ip_address)

    elif mapped_action["name"] == "SQLInjection":
        action = SQLInjection(session=session, agent = "Red", target_session = target_session, ip_address=ip_address)

    elif mapped_action["name"] == "SSHBruteForce":
        action = SSHBruteForce(session=session, agent = "Red", target_session = target_session, ip_address=ip_address)

    else:
        print("The action name in the dictionary does not correspond to any existing Cyborg action.")

    return action


class BlueTable():
    def __init__(self, init_obs, last_action = None):
        self.baseline = None
        self.blue_info = {}
        self.last_action = last_action
        self.info = None
        self._process_initial_obs(init_obs)
        self.observation_change(init_obs, last_action = last_action, baseline=True)


    def _process_initial_obs(self, obs):
        obs = obs.copy()
        self.baseline = obs
        del self.baseline['success']
        for hostid in obs:
            if hostid == 'success':
                continue
            host = obs[hostid]
            interface = host['Interface'][0]
            subnet = interface['Subnet']
            ip = str(interface['IP Address'])
            hostname = host['System info']['Hostname']
            self.blue_info[hostname] = [str(subnet),str(ip),hostname, 'None','No']
        return self.blue_info

    def observation_change(self,observation, last_action, baseline=False):
        self.last_action = last_action
        obs = observation if type(observation) == dict else observation.data
        obs = deepcopy(observation)
        success = obs['success']

        self._process_last_action()
        anomaly_obs = self._detect_anomalies(obs) if not baseline else obs
        del obs['success']
        # TODO check what info is for baseline
        info = self._process_anomalies(anomaly_obs)
        if baseline:
            for host in info:
                info[host][-2] = 'None'
                info[host][-1] = 'No'
                self.blue_info[host][-1] = 'No'

        self.info = info

        return self._create_vector(success)

    def _process_last_action(self):
        action = self.last_action
        if action is not None:
            name = action.__class__.__name__
            hostname = action.get_params()['hostname'] if name in ('Restore','Remove') else None

            if name == 'Restore':
                self.blue_info[hostname][-1] = 'No'
            elif name == 'Remove':
                compromised = self.blue_info[hostname][-1]
                if compromised != 'No':
                    self.blue_info[hostname][-1] = 'Unknown'

    def _detect_anomalies(self,obs):
        if self.baseline is None:
            raise TypeError('BlueTableWrapper was unable to establish baseline. This usually means the environment was not reset before calling the step method.')

        anomaly_dict = {}

        for hostid,host in obs.items():
            if hostid == 'success':
                continue

            host_baseline = self.baseline[hostid]
            if host == host_baseline:
                continue

            host_anomalies = {}
            if 'Files' in host:
                baseline_files = host_baseline.get('Files',[])
                anomalous_files = []
                for f in host['Files']:
                    if f not in baseline_files:
                        anomalous_files.append(f)
                if anomalous_files:
                    host_anomalies['Files'] = anomalous_files

            if 'Processes' in host:
                baseline_processes = host_baseline.get('Processes',[])
                anomalous_processes = []
                for p in host['Processes']:
                    if p not in baseline_processes:
                        anomalous_processes.append(p)
                if anomalous_processes:
                    host_anomalies['Processes'] = anomalous_processes

            if host_anomalies:
                anomaly_dict[hostid] = host_anomalies

        return anomaly_dict

    def _process_anomalies(self,anomaly_dict):
        info = deepcopy(self.blue_info)
        for hostid, host_anomalies in anomaly_dict.items():
            assert len(host_anomalies) > 0
            if 'Processes' in host_anomalies:
                connection_type = self._interpret_connections(host_anomalies['Processes'])
                info[hostid][-2] = connection_type
                if connection_type == 'Exploit':
                    info[hostid][-1] = 'User'
                    self.blue_info[hostid][-1] = 'User'
            if 'Files' in host_anomalies:
                malware = [f['Density'] >= 0.9 for f in host_anomalies['Files']]
                if any(malware):
                    info[hostid][-1] = 'Privileged'
                    self.blue_info[hostid][-1] = 'Privileged'

        return info

    def _interpret_connections(self,activity:list):
        num_connections = len(activity)

        ports = set([item['Connections'][0]['local_port'] \
            for item in activity if 'Connections' in item])
        port_focus = len(ports)

        remote_ports = set([item['Connections'][0].get('remote_port') \
            for item in activity if 'Connections' in item])
        if None in remote_ports:
            remote_ports.remove(None)

        if num_connections >= 3 and port_focus >=3:
            anomaly = 'Scan'
        elif 4444 in remote_ports:
            anomaly = 'Exploit'
        elif num_connections >= 3 and port_focus == 1:
            anomaly = 'Exploit'
        elif 'Service Name' in activity[0]:
            anomaly = 'None'
        else:
            anomaly = 'Scan'

        return anomaly


    def _create_blue_table(self, success):
        table = PrettyTable([
            'Subnet',
            'IP Address',
            'Hostname',
            'Activity',
            'Compromised'
            ])
        for hostid in self.info:
            table.add_row(self.info[hostid])

        table.sortby = 'Hostname'
        table.success = success
        return table

    def _create_vector(self, success):
        table = self._create_blue_table(success)._rows

        proto_vector = []
        for row in table:
            # Activity
            activity = row[3]
            if activity == 'None':
                value = [0,0]
            elif activity == 'Scan':
                value = [1,0]
            elif activity == 'Exploit':
                value = [1,1]
            else:
                raise ValueError('Table had invalid Access Level')
            proto_vector.extend(value)

            # Compromised
            compromised = row[4]
            if compromised == 'No':
                value = [0, 0]
            elif compromised == 'Unknown':
                value = [1, 0]
            elif compromised == 'User':
                value = [0,1]
            elif compromised == 'Privileged':
                value = [1,1]
            else:
                raise ValueError('Table had invalid Access Level')
            proto_vector.extend(value)

        return np.array(proto_vector)
