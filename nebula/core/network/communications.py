import asyncio
import collections
import logging
import json  # Importado a nivel global para eficiencia
import hashlib # Importado a nivel global para eficiencia
from typing import TYPE_CHECKING

import requests

from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import MessageEvent, DuplicatedMessageEvent, RoundStartEvent
from nebula.core.network.blacklist import BlackList
from nebula.core.network.connection import Connection
from nebula.core.network.discoverer import Discoverer
from nebula.core.network.externalconnection.externalconnectionservice import factory_connection_service
from nebula.core.network.forwarder import Forwarder
from nebula.core.network.messages import MessagesManager
from nebula.core.network.propagator import Propagator
from nebula.core.utils.locker import Locker

if TYPE_CHECKING:
    from nebula.core.engine import Engine

BLACKLIST_EXPIRATION_TIME = 60

_COMPRESSED_MESSAGES = ["model", "offer_model"]


class CommunicationsManager:
    # --- DEDUPLICACIÓN DE FLOODING ---
    # Diccionarios para almacenar hashes de mensajes procesados por ronda
    _processed_reputation_floods = {}
    _processed_topology_floods = {}

    def register_reputation_flood_callbacks(self):
        """
        Registra los callbacks para flooding de reputación y SINCRONIZACIÓN.
        """
        from nebula.core.eventmanager import EventManager
        from nebula.core.nebulaevents import RoundStartEvent

        # Callbacks de reputación
        asyncio.create_task(EventManager.get_instance().subscribe_node_event(RoundStartEvent, self.on_round_start_reputation_flood))
        asyncio.create_task(EventManager.get_instance().subscribe(("control", "reputation_flood"), self.handle_reputation_flood_message))

        # --- NUEVO: Callback de Sincronización ---
        # Usamos "alive" porque es seguro en protobuf (Action 0) y filtramos por payload
        asyncio.create_task(EventManager.get_instance().subscribe(("control", "alive"), self.handle_sync_round_message))    # --- NUEVOS MÉTODOS PARA SINCRONIZACIÓN ---

    async def send_sync_signal(self, specific_round=None):
        """
        Envía un mensaje con la RONDA ACTUAL explícita a los vecinos.
        """
        try:
            # Obtenemos la ronda actual del motor (o usamos la específica si se provee para evitar locks)
            if specific_round is not None:
                current_round = specific_round
            else:
                current_round = await self.get_round()

            # Payload explícito con la ronda
            payload = json.dumps({"type": "sync", "round": current_round})

            # Usamos "alive" para evitar problemas con Enums de Protobuf
            msg = self.create_message("control", "alive", log=payload)

            # Enviar a todos los vecinos conectados (incluso si no son directos, por si acaso)
            # Usamos send_message para no bloquear
            neighbors = list(self.connections.keys())
            for neighbor in neighbors:
                asyncio.create_task(self.send_message(neighbor, msg))

        except Exception as e:
            logging.error(f"[SYNC] Error sending sync signal: {e}")

    async def handle_sync_round_message(self, source, message):
        """
        Recibe el SYNC, lee la ronda del JSON y FUERZA la actualización de la conexión.
        """
        try:
            if not message or not hasattr(message, 'log'):
                return

            # Si no es un mensaje JSON o falla el decode, asumimos que es un alive normal y salimos
            try:
                data = json.loads(message.log)
            except:
                return

            if data.get("type") != "sync":
                return

            peer_round = data.get("round", -1)

            # Si el nodo está en mis conexiones, actualizo su estado manualmente
            if source in self.connections:
                conn = self.connections[source]

                # Solo actualizamos si la ronda reportada es mayor que la que tenemos guardada
                if peer_round > conn.round:
                    old_round = conn.round
                    conn.round = peer_round # <--- AQUÍ ESTÁ LA CLAVE DEL DESBLOQUEO
                    logging.info(f"🔄 [SYNC] Neighbor {source} updated: R{old_round} -> R{peer_round}")

        except Exception as e:
            logging.warning(f"[SYNC] Error processing sync message from {source}: {e}")

    async def on_round_start_reputation_flood(self, event):
        current_round = getattr(event, 'round', -1)
        logging.info(f"[REPUTATION][DEBUG] Inicio de ronda {current_round}. Limpiando cachés de flooding.")

        # --- FIX 1: Limpieza de caché para evitar saturación de memoria ---
        self._processed_reputation_floods.clear()
        self._processed_topology_floods.clear()
        # ----------------------------------------------------------------

        logging.info(f"[REPUTATION] Broadcast GLOBAL reputation to ALL known nodes (round {current_round})")
        reputation = None
        if hasattr(self._engine, '_reputation') and self._engine._reputation:
            try:
                reputation = self._engine._reputation.get_reputation_table()
            except Exception as e:
                logging.warning(f"[REPUTATION] Error getting reputation table: {e}")

        if not reputation:
            reputation = {}

        # Estructura del mensaje con metadatos de ronda
        payload = {"type": "reputation_flood", "data": reputation, "round": current_round}

        # --- FIX 2: sort_keys=True para que el hash sea idéntico en todos los nodos ---
        rep_json = json.dumps(payload, sort_keys=True)

        message = self.create_message("control", "reputation_flood", log=rep_json)

        # Enviar a vecinos directos para iniciar el chisme (Gossip)
        direct_neighbors = set(self.connections.keys())
        for node in direct_neighbors:
            asyncio.create_task(self.send_message(node, message))

    async def handle_reputation_flood_message(self, message, payload):
        """
        Handles the reception of a reputation flood message with Gossip Forwarding.
        """
        if not message or not payload: return

        # 1. Deduplication (Hash content to avoid loops)
        try:
            content_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        except: return

        if content_hash in self._processed_reputation_floods:
            return

        # Mark as processed
        self._processed_reputation_floods[content_hash] = time.time()

        # 2. Process Data (Update Local View with Source Context)
        try:
             sender_id = message.source_server if hasattr(message, 'source_server') else "unknown"
             if "data" in payload:
                self._merge_global_reputation(payload["data"], sender_id)
        except Exception as e:
            self.logger.error(f"Error merging reputation: {e}")

        # 3. Forward to Neighbors (Gossip) -> Skip Sender
        sender = message.source_server
        original_log = message.log # Use original JSON to preserve exact hash

        neighbors = []
        if hasattr(self, 'connections'): neighbors = list(self.connections.keys())

        for neighbor in neighbors:
            if neighbor == sender: continue

            # Create Wrapper Message
            fwd_msg = self.create_message("control", "reputation_flood", log=original_log)
            asyncio.create_task(self.send_message(neighbor, fwd_msg))

    def _merge_global_reputation(self, external_data, sender_id="unknown"):
        """
        Merges external reputation data with local data.
        """
        if not self._engine._reputation:
            return

        # Try to use intelligent merge if available
        if hasattr(self._engine._reputation, "process_external_feedback"):
             self._engine._reputation.process_external_feedback(sender_id, external_data)
        elif hasattr(self._engine._reputation, "reputation_table"):
            # Fallback for simple tables
            # WARNING: This overwrites local view with flood data if not handled carefully.
            # Ideally we rely on RepModule. Since we don't have process_external_feedback widely,
            # we just log it or update if valid.
            pass

    def register_topology_flood_callbacks(self):
        from nebula.core.nebulaevents import RoundStartEvent
        asyncio.create_task(EventManager.get_instance().subscribe_node_event(RoundStartEvent, self.on_round_start_topology_flood))
        # Usamos topic específico
        asyncio.create_task(EventManager.get_instance().subscribe(("control", "topology_flood"), self.handle_topology_flood_message))

    async def on_round_start_topology_flood(self, event):
        current_round = getattr(event, 'round', -1)

        if hasattr(self, '_global_topology') and self._global_topology:
            topology = {node: list(neighs) for node, neighs in self._global_topology.items()}
        else:
            topology = self._get_local_topology()

        payload = {"type": "topology_flood", "data": topology, "round": current_round}
        # Hashing determinista
        topology_json = json.dumps(payload, sort_keys=True)

        message = self.create_message("control", "topology_flood", log=topology_json)
        await self.send_message_to_neighbors(message)

    async def handle_topology_flood_message(self, source, message):
        if not message or not hasattr(message, 'log'): return
        try:
            payload = json.loads(message.log)
        except: return

        if not isinstance(payload, dict) or payload.get("type") != "topology_flood":
            return

        current_round = await self.get_round()
        msg_round = payload.get("round", -1)

        # Filtro de ronda
        if current_round is not None and msg_round < current_round - 1:
            return

        # Deduplicación
        canonical_content = json.dumps(payload, sort_keys=True)
        hash_val = hashlib.sha256(canonical_content.encode()).hexdigest()

        if hash_val in self._processed_topology_floods:
            return

        self._processed_topology_floods[hash_val] = True

        # Merge
        self._merge_global_topology(payload.get("data", {}))

        # Forward
        neighbors = set(self.connections.keys())
        if source in neighbors: neighbors.discard(source)

        fwd_message = self.create_message("control", "topology_flood", log=canonical_content)
        for neighbor in neighbors:
            asyncio.create_task(self.send_message(neighbor, fwd_message))

    def _get_local_topology(self):
        return {self.addr: list(self.connections.keys())}

    def _merge_global_topology(self, incoming_topology):
        if not hasattr(self, '_global_topology'):
            self._global_topology = {}
        for node, neighbors in incoming_topology.items():
            if node not in self._global_topology:
                self._global_topology[node] = set(neighbors)
            else:
                self._global_topology[node].update(neighbors)

        local_neighbors = set(self.connections.keys())
        if self.addr not in self._global_topology:
            self._global_topology[self.addr] = local_neighbors
        else:
            self._global_topology[self.addr].update(local_neighbors)

    def get_global_topology(self):
        if hasattr(self, '_global_topology'):
            return {node: list(neighs) for node, neighs in self._global_topology.items()}
        else:
            return self._get_local_topology()

    """
    Singleton class responsible for managing all communications in the Nebula system.
    """

    _instance = None
    _lock = Locker("communications_manager_lock", async_lock=False)

    def __new__(cls, engine: "Engine"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise ValueError("CommunicationsManager has not been initialized yet.")
        return cls._instance

    def __init__(self, engine: "Engine"):
        if hasattr(self, "_initialized") and self._initialized:
            return

        logging.info("🌐  Initializing Communications Manager")
        self._engine = engine
        self.addr = engine.get_addr()
        self.host = self.addr.split(":")[0]
        self.port = int(self.addr.split(":")[1])
        self.config = engine.get_config()
        self.id = str(self.config.participant["device_args"]["idx"])

        self.register_endpoint = f"http://{self.config.participant['scenario_args']['controller']}/platform/dashboard/{self.config.participant['scenario_args']['name']}/node/register"
        self.wait_endpoint = f"http://{self.config.participant['scenario_args']['controller']}/platform/dashboard/{self.config.participant['scenario_args']['name']}/node/wait"

        self._connections: dict[str, Connection] = {}
        self.connections_lock = Locker(name="connections_lock", async_lock=True)
        self.connections_manager_lock = Locker(name="connections_manager_lock", async_lock=True)
        self.connection_attempt_lock_incoming = Locker(name="connection_attempt_lock_incoming", async_lock=True)
        self.connection_attempt_lock_outgoing = Locker(name="connection_attempt_lock_outgoing", async_lock=True)

        self.pending_connections = set()
        self.incoming_connections = {}
        self.outgoing_connections = {}
        self.ready_connections = set()
        self._ready_connections_lock = Locker("ready_connections_lock", async_lock=True)

        self._mm = MessagesManager(addr=self.addr, config=self.config)
        self.received_messages_hashes = collections.deque(
            maxlen=self.config.participant["message_args"]["max_local_messages"]
        )
        self.receive_messages_lock = Locker(name="receive_messages_lock", async_lock=True)

        self._discoverer = Discoverer(addr=self.addr, config=self.config)
        self._health = None
        self._forwarder = Forwarder(config=self.config)
        self._propagator = Propagator()

        self.connections_reconnect = []
        self.max_connections = 1000
        self.network_engine = None

        self.stop_network_engine = asyncio.Event()
        self.loop = asyncio.get_event_loop()
        max_concurrent_tasks = 5
        self.semaphore_send_model = asyncio.Semaphore(max_concurrent_tasks)

        self._blacklist = BlackList()
        self._external_connection_service = factory_connection_service("nebula", self.addr)

        self._initialized = True
        self._running = asyncio.Event()
        logging.info("Communication Manager initialization completed")

    @property
    def engine(self): return self._engine
    @property
    def connections(self): return self._connections
    @property
    def mm(self): return self._mm
    @property
    def discoverer(self): return self._discoverer
    @property
    def health(self): return self._health
    @property
    def forwarder(self): return self._forwarder
    @property
    def propagator(self): return self._propagator
    @property
    def ecs(self): return self._external_connection_service
    @property
    def bl(self): return self._blacklist

    async def check_federation_ready(self):
        logging.info(f"🔗  check_federation_ready | Ready connections: {self.ready_connections} | Connections: {self.connections.keys()}")
        async with self.connections_lock:
            async with self._ready_connections_lock:
                if set(self.connections.keys()) == self.ready_connections:
                    return True

    async def add_ready_connection(self, addr):
        async with self._ready_connections_lock:
            self.ready_connections.add(addr)

    async def start_communications(self, initial_neighbors):
        self._running.set()
        logging.info(f"Neighbors: {self.config.participant['network_args']['neighbors']}")
        logging.info(f"💤  Cold start time: {self.config.participant['misc_args']['grace_time_connection']} seconds before connecting to the network")
        await asyncio.sleep(self.config.participant["misc_args"]["grace_time_connection"])
        await self.start()
        neighbors = set(initial_neighbors)
        if self.addr in neighbors: neighbors.discard(self.addr)

        for addr in neighbors:
            await self.connect(addr, direct=True)
            await asyncio.sleep(1)
        while not await self.verify_connections(neighbors):
            await asyncio.sleep(1)
        current_connections = await self.get_addrs_current_connections()
        logging.info(f"Connections verified: {current_connections}")
        await self.deploy_additional_services()

    async def handle_incoming_message(self, data, addr_from):
        if not await self.bl.node_in_blacklist(addr_from):
            await self.mm.process_message(data, addr_from)

    async def forward_message(self, data, addr_from):
        logging.info("Forwarding message... ")
        await self.forwarder.forward(data, addr_from=addr_from)

    async def handle_message(self, message_event):
        asyncio.create_task(EventManager.get_instance().publish(message_event))

    async def handle_model_message(self, source, message):
        logging.info(f"🤖  handle_model_message | Received model from {source} with round {message.round}")

        # --- FIX: Update neighbor round based on Model Message ---
        # This ensures the Engine knows the neighbor has advanced without waiting for a SYNC signal
        try:
            if source in self.connections:
                conn = self.connections[source]
                if message.round > conn.round:
                    logging.info(f"🔄 [Implicit Sync] Neighbor {source} updated via Model: R{conn.round} -> R{message.round}")
                    conn.round = message.round
        except Exception as e:
            logging.warning(f"Error updating neighbor round from model message: {e}")
        # ---------------------------------------------------------

        if message.round == -1:
            model_init_event = MessageEvent(("model", "initialization"), source, message)
            asyncio.create_task(EventManager.get_instance().publish(model_init_event))
        else:
            model_updt_event = MessageEvent(("model", "update"), source, message)
            asyncio.create_task(EventManager.get_instance().publish(model_updt_event))

    def create_message(self, message_type: str, action: str = "", *args, **kwargs):
        return self.mm.create_message(message_type, action, *args, **kwargs)

    def get_messages_events(self):
        return self.mm.get_messages_events()

    async def add_to_recently_disconnected(self, addr): await self.bl.add_recently_disconnected(addr)
    async def add_to_blacklist(self, addr): await self.bl.add_to_blacklist(addr)
    async def get_blacklist(self): return await self.bl.get_blacklist()
    async def apply_restrictions(self, nodes: set) -> set | None: return await self.bl.apply_restrictions(nodes)
    async def clear_restrictions(self): await self.bl.clear_restrictions()

    async def start_external_connection_service(self, run_service=True):
        if self.ecs == None: self._external_connection_service = factory_connection_service(self, self.addr)
        if run_service: await self.ecs.start()
    async def stop_external_connection_service(self): await self.ecs.stop()
    async def init_external_connection_service(self): await self.start_external_connection_service()
    async def is_external_connection_service_running(self): return await self.ecs.is_running()
    async def start_beacon(self): await self.ecs.start_beacon()
    async def stop_beacon(self): await self.ecs.stop_beacon()
    async def modify_beacon_frequency(self, frequency): await self.ecs.modify_beacon_frequency(frequency)

    async def stablish_connection_to_federation(self, msg_type="discover_join", addrs_known=None) -> tuple[int, set]:
        addrs = []
        if addrs_known == None:
            logging.info("Searching federation process beginning...")
            addrs = await self.ecs.find_federation()
            logging.info(f"Found federation devices | addrs {addrs}")
        else:
            logging.info(f"Searching federation process beginning... | Using addrs previously known {addrs_known}")
            addrs = addrs_known

        msg = self.create_message("discover", msg_type)
        neighbors = await self.get_addrs_current_connections(only_direct=True, myself=True)
        addrs = set(addrs)
        if neighbors: addrs.difference_update(neighbors)

        discovers_sent = 0
        connections_made = set()
        if addrs:
            logging.info("Starting communications with devices found")
            max_tries = 5
            for addr in addrs:
                await self.connect(addr, direct=False, priority="high")
                connections_made.add(addr)
                await asyncio.sleep(1)
            for i in range(0, max_tries):
                if await self.verify_any_connections(addrs): break
                await asyncio.sleep(1)
            current_connections = await self.get_addrs_current_connections(only_undirected=True)
            logging.info(f"Connections verified after searching: {current_connections}")

            for addr in addrs:
                logging.info(f"Sending {msg_type} to addr: {addr}")
                asyncio.create_task(self.send_message(addr, msg))
                await asyncio.sleep(1)
                discovers_sent += 1
        return (discovers_sent, connections_made)

    def get_connections_lock(self): return self.connections_lock
    def get_config(self): return self.config
    def get_addr(self): return self.addr
    async def get_round(self): return await self.engine.get_round()

    async def start(self):
        """
        Starts the communications manager by deploying the network engine to accept incoming connections.
        """
        logging.info("🌐  Starting Communications Manager...")

        # --- NUEVO: REGISTRAR CALLBACKS AQUÍ PARA ASEGURAR QUE FUNCIONAN ---
        self.register_reputation_flood_callbacks()
        self.register_topology_flood_callbacks()
        # -------------------------------------------------------------------

        await self.deploy_network_engine()

    async def deploy_network_engine(self):
        logging.info("🌐  Deploying Network engine...")
        self.network_engine = await asyncio.start_server(self.handle_connection_wrapper, self.host, self.port)
        self.network_task = asyncio.create_task(self.network_engine.serve_forever(), name="Network Engine")
        logging.info(f"🌐  Network engine deployed at host {self.host} and port {self.port}")

    async def handle_connection_wrapper(self, reader, writer):
        asyncio.create_task(self.handle_connection(reader, writer))

    async def handle_connection(self, reader, writer, priority="medium"):
        async def process_connection(reader, writer, priority="medium"):
            try:
                addr = writer.get_extra_info("peername")
                if await self.engine.learning_cycle_finished():
                    writer.write(b"CONNECTION//CLOSE\n")
                    await writer.drain()
                    writer.close()
                    await writer.wait_closed()
                    return

                connected_node_id = await reader.readline()
                connected_node_id = connected_node_id.decode("utf-8").strip()
                connected_node_port = addr[1]
                if ":" in connected_node_id:
                    connected_node_id, connected_node_port = connected_node_id.split(":")
                connection_addr = f"{addr[0]}:{connected_node_port}"
                direct = await reader.readline()
                direct = direct.decode("utf-8").strip()
                direct = direct == "True"

                blacklist = await self.bl.get_blacklist()
                if blacklist and connection_addr in blacklist:
                    writer.close()
                    await writer.wait_closed()
                    return

                if self.id == connected_node_id:
                    writer.write(b"CONNECTION//CLOSE\n")
                    await writer.drain()
                    writer.close()
                    await writer.wait_closed()
                    return

                async with self.connections_manager_lock:
                    async with self.connections_lock:
                        if len(self.connections) >= self.max_connections:
                            writer.write(b"CONNECTION//CLOSE\n")
                            await writer.drain()
                            writer.close()
                            await writer.wait_closed()
                            return

                        if connection_addr in self.connections:
                            writer.write(b"CONNECTION//EXISTS\n")
                            await writer.drain()
                            writer.close()
                            await writer.wait_closed()
                            return

                    if connection_addr in self.pending_connections:
                        if int(self.host.split(".")[3]) < int(addr[0].split(".")[3]):
                            writer.write(b"CONNECTION//CLOSE\n")
                            await writer.drain()
                            writer.close()
                            await writer.wait_closed()
                            return
                        else:
                            if connection_addr in self.outgoing_connections:
                                out_reader, out_writer = self.outgoing_connections.pop(connection_addr)
                                out_writer.write(b"CONNECTION//CLOSE\n")
                                await out_writer.drain()
                                out_writer.close()
                                await out_writer.wait_closed()

                    self.pending_connections.add(connection_addr)
                    self.incoming_connections[connection_addr] = (reader, writer)

                logging.info(f"🔗  [incoming] Creating new connection with {addr} (id {connected_node_id})")
                await writer.drain()
                connection = Connection(reader, writer, connected_node_id, addr[0], connected_node_port, direct=direct, config=self.config, prio=priority)
                async with self.connections_manager_lock:
                    async with self.connections_lock:
                        self.connections[connection_addr] = connection
                        writer.write(b"CONNECTION//NEW\n")
                        await writer.drain()
                        writer.write(f"{self.id}\n".encode())
                        await writer.drain()
                        await connection.start()

            except Exception as e:
                logging.exception(f"❗️  [incoming] Error while handling connection with {addr}: {e}")
            finally:
                if connection_addr in self.pending_connections: self.pending_connections.remove(connection_addr)
                if connection_addr in self.incoming_connections: self.incoming_connections.pop(connection_addr)

        await process_connection(reader, writer, priority)

    async def terminate_failed_reconnection(self, conn: Connection):
        connected_with = conn.addr
        await self.bl.add_recently_disconnected(connected_with)
        await self.disconnect(connected_with, mutual_disconnection=False)

    async def stop(self):
        logging.info("🌐  Stopping Communications Manager...")
        if self.network_engine:
            self.network_engine.close()
            await self.network_engine.wait_closed()
            if hasattr(self, "network_task") and self.network_task:
                self.network_task.cancel()
                try: await self.network_task
                except asyncio.CancelledError: pass

        async with self.connections_lock:
            connections = list(self.connections.values())
            for node in connections: await node.stop()

        if self._forwarder: await self._forwarder.stop()
        if self.ecs: await self.ecs.stop()
        if self.discoverer: await self.discoverer.stop()
        if self.health:
            try: await self.health.stop()
            except: pass
        if self._propagator: await self._propagator.stop()
        if self._blacklist: await self._blacklist.stop()

        self._running.clear()
        self.stop_network_engine.set()
        logging.info("🌐  Communications Manager stopped successfully")

    async def run_reconnections(self):
        for connection in self.connections_reconnect:
            if connection["addr"] in self.connections:
                connection["tries"] = 0
            else:
                connection["tries"] += 1
                await self.connect(connection["addr"])

    async def clear_unused_undirect_connections(self):
        async with self.connections_lock:
            inactive_connections = [conn for conn in self.connections.values() if await conn.is_inactive()]
        for conn in inactive_connections:
            asyncio.create_task(self.disconnect(conn.addr, mutual_disconnection=False))

    async def verify_any_connections(self, neighbors):
        async with self.connections_lock:
            if any(neighbor in self.connections for neighbor in neighbors): return True
            return False

    async def verify_connections(self, neighbors):
        async with self.connections_lock:
            return bool(all(neighbor in self.connections for neighbor in neighbors))

    async def network_wait(self): await self.stop_network_engine.wait()

    async def deploy_additional_services(self):
        logging.info("🌐  Deploying additional services...")
        await self._forwarder.start()
        await self._propagator.start()

    async def include_received_message_hash(self, hash_message, source):
        try:
            await self.receive_messages_lock.acquire_async()
            if hash_message in self.received_messages_hashes:
                duplicated_event = DuplicatedMessageEvent(source, "Duplicated message received")
                asyncio.create_task(EventManager.get_instance().publish_node_event(duplicated_event))
                return False
            self.received_messages_hashes.append(hash_message)
            return True
        except Exception as e:
            logging.exception(f"❗️  handle_incoming_message | Error including message hash: {e}")
            return False
        finally:
            await self.receive_messages_lock.release_async()

    async def send_message_to_neighbors(self, message, neighbors=None, interval=0):
        if neighbors is None:
            current_connections = await self.get_all_addrs_current_connections(only_direct=True)
            neighbors = set(current_connections)
        for neighbor in neighbors:
            asyncio.create_task(self.send_message(neighbor, message))
            if interval > 0: await asyncio.sleep(interval)

    async def send_message(self, dest_addr, message, message_type=""):
        is_compressed = message_type in _COMPRESSED_MESSAGES
        if not is_compressed:
            try:
                if dest_addr in self.connections:
                    conn = self.connections[dest_addr]
                    await conn.send(data=message)
            except Exception as e:
                logging.exception(f"❗️  Cannot send message {message} to {dest_addr}. Error: {e!s}")
                await self.disconnect(dest_addr, mutual_disconnection=False)
        else:
            async with self.semaphore_send_model:
                try:
                    conn = self.connections.get(dest_addr)
                    if conn is None: return
                    await conn.send(data=message, is_compressed=True)
                except Exception as e:
                    logging.exception(f"❗️  Cannot send model to {dest_addr}: {e!s}")
                    await self.disconnect(dest_addr, mutual_disconnection=False)

    async def establish_connection(self, addr, direct=True, reconnect=False, priority="medium"):
        if await self.engine.learning_cycle_finished(): return False
        logging.info(f"🔗  [outgoing] Establishing connection with {addr} (direct: {direct})")

        async def process_establish_connection(addr, direct, reconnect, priority):
            try:
                host = str(addr.split(":")[0])
                port = str(addr.split(":")[1])
                if host == self.host and port == self.port: return False

                blacklist = await self.bl.get_blacklist()
                if blacklist and addr in blacklist: return

                async with self.connections_manager_lock:
                    async with self.connections_lock:
                        if addr in self.connections:
                            if not self.connections[addr].get_direct() and (direct == True):
                                self.connections[addr].set_direct(direct)
                                return True
                            else: return False
                    if addr in self.pending_connections:
                        if int(self.host.split(".")[3]) >= int(host.split(".")[3]): return False
                        else:
                            if addr in self.incoming_connections:
                                inc_reader, inc_writer = self.incoming_connections.pop(addr)
                                inc_writer.write(b"CONNECTION//CLOSE\n")
                                await inc_writer.drain()
                                inc_writer.close()
                                await inc_writer.wait_closed()
                    self.pending_connections.add(addr)

                reader, writer = await asyncio.open_connection(host, port)
                async with self.connections_manager_lock: self.outgoing_connections[addr] = (reader, writer)

                writer.write(f"{self.id}:{self.port}\n".encode())
                await writer.drain()
                writer.write(f"{direct}\n".encode())
                await writer.drain()

                connection_status = await reader.readline()
                connection_status = connection_status.decode("utf-8").strip()

                if connection_status == "CONNECTION//CLOSE":
                    writer.close()
                    await writer.wait_closed()
                    return False
                elif connection_status == "CONNECTION//PENDING":
                    writer.close()
                    await writer.wait_closed()
                    return False
                elif connection_status == "CONNECTION//EXISTS":
                    writer.close()
                    await writer.wait_closed()
                    return True
                elif connection_status == "CONNECTION//NEW":
                    async with self.connections_manager_lock:
                        connected_node_id = await reader.readline()
                        connected_node_id = connected_node_id.decode("utf-8").strip()
                        connection = Connection(reader, writer, connected_node_id, host, port, direct=direct, config=self.config, prio=priority)
                        async with self.connections_lock: self.connections[addr] = connection
                        await connection.start()
                else:
                    writer.close()
                    await writer.wait_closed()
                    return False

                if reconnect: self.connections_reconnect.append({"addr": addr, "tries": 0})
                if direct: self.config.add_neighbor_from_config(addr)
                return True
            except Exception as e:
                logging.info(f"❗️  [outgoing] Error adding direct connected neighbor {addr}: {e!s}")
                return False
            finally:
                if addr in self.pending_connections: self.pending_connections.remove(addr)
                if addr in self.outgoing_connections: self.outgoing_connections.pop(addr)
                if addr in self.incoming_connections: self.incoming_connections.pop(addr)

        asyncio.create_task(process_establish_connection(addr, direct, reconnect, priority))

    async def connect(self, addr, direct=True, priority="medium"):
        async with self.connections_lock: duplicated = addr in self.connections
        if duplicated:
            if direct:
                if not self.connections[addr].get_direct():
                    return await self.establish_connection(addr, direct=True, reconnect=False, priority=priority)
                else: return await self.establish_connection(addr, direct=True, reconnect=False, priority=priority)
            else: return False
        else:
            return await self.establish_connection(addr, direct=direct, reconnect=False, priority=priority)

    async def register(self):
        data = {"node": self.addr}
        try:
            response = requests.post(self.register_endpoint, json=data)
            if response.status_code == 200: logging.info(f"Node {self.addr} registered successfully")
            else: logging.error(f"Error registering node {self.addr}")
        except: pass

    async def wait_for_controller(self):
        while await self.is_running():
            try:
                response = requests.get(self.wait_endpoint)
                if response.status_code == 200: break
            except: pass
            await asyncio.sleep(1)

    async def is_running(self): return self._running.is_set()

    async def disconnect(self, dest_addr, mutual_disconnection=True, forced=False):
        logging.info(f"Trying to disconnect {dest_addr}")
        is_neighbor = dest_addr in await self.get_addrs_current_connections(only_direct=True, myself=True)
        if forced: await self.add_to_blacklist(dest_addr)

        async with self.connections_lock:
            connection_to_remove = self.connections.get(dest_addr)
            if not connection_to_remove: return
            conn = self.connections[dest_addr]

        try:
            if mutual_disconnection:
                try:
                    await conn.send(data=self.create_message("connection", "disconnect"))
                    async with self.connections_lock:
                        if dest_addr in self.connections: self.connections.pop(dest_addr)
                    await conn.stop()
                except Exception as e:
                    async with self.connections_lock:
                        if dest_addr in self.connections: self.connections.pop(dest_addr)
                    await conn.stop()
            else:
                async with self.connections_lock:
                    if dest_addr in self.connections: self.connections.pop(dest_addr)
                await conn.stop()

            current_connections = await self.get_all_addrs_current_connections(only_direct=True)
            self.config.update_neighbors_from_config(set(current_connections), dest_addr)
            if is_neighbor:
                current_connections = await self.get_addrs_current_connections(only_direct=True, myself=True)
                await self.engine.update_neighbors(dest_addr, current_connections, remove=True)

        except Exception as e:
            logging.exception(f"Error during disconnection of {dest_addr}: {e!s}")
            async with self.connections_lock:
                if dest_addr in self.connections: self.connections.pop(dest_addr)
            try: await conn.stop()
            except: pass
            raise

    async def get_all_addrs_current_connections(self, only_direct=False, only_undirected=False):
        try:
            await self.get_connections_lock().acquire_async()
            if only_direct: return {addr for addr, conn in self.connections.items() if conn.get_direct()}
            elif only_undirected: return {addr for addr, conn in self.connections.items() if not conn.get_direct()}
            else: return set(self.connections.keys())
        finally: await self.get_connections_lock().release_async()

    async def get_addrs_current_connections(self, only_direct=False, only_undirected=False, myself=False):
        current_connections = await self.get_all_addrs_current_connections(only_direct=only_direct, only_undirected=only_undirected)
        current_connections = set(current_connections)
        if myself: current_connections.add(self.addr)
        return current_connections

    def get_ready_connections(self): return {addr for addr, conn in self.connections.items() if conn.get_ready()}
    async def learning_finished(self): return await self.engine.learning_cycle_finished()
    def __str__(self): return f"Connections: {[str(conn) for conn in self.connections.values()]}"
