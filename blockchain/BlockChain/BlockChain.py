import hashlib as hasher
import json
from time import time
from textwrap import dedent
from uuid import uuid4
from flask import Flask, jsonify, request

from urllib.parse import urlparse

#管理链式数据
class BlockChain(object):


    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.nodes = set()
        #初始块




    def register_node(self,address):
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)

    def valid_chain(self,chain):
        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            print(f'{last_block}')
            print(f'{block}')
            print("\n----------------\n")
            # check hash
            if block['prev_hash'] != self.hash(last_block):
                return False
            # check proof
            if not self.valid_proof(last_block['proof'],block['proof']):
                return False
            last_block = block
            current_index += 1

        return True

    def resolve_conflicts(self):
        """
        This is our Consensus Algorithm, it resolves conflicts
        by replacing our chain with the longest one in the network.
        :return: <bool> True if our chain was replaced, False if not
        """
        neighbours = self.nodes
        new_chain = None

        max_length = len(self.chain)

        for node in neighbours:
            response = request.get(f'http://{node}/chain')

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
        if new_chain:
            self.chain = new_chain
            return True

        return False

    def new_block(self,proof,pre_hash):
        #create new block and add to chain
        block = {
            'index':len(self.chain)+1,
            'timestamp':time(),
            'transactions':self.current_transactions,
            'proof':proof,
            'prev_hash':pre_hash or self.hash(self.chain[-1]),
        }
        self.current_transactions = []
        self.chain.append(block)
        return block

    def new_transactions(self,sender,recipient,amount):
        """
        add new transaction to transactions
        sender:str address of sender 发送者
        recipient: str address of 接收者
        amount ：数据
        返回交易被添加到区块的索引
        """
        self.current_transaction.append({
            'sender':sender,
            'recipient':recipient,
            'amount':amount,
        })

        return self.last_block['index']+1

    def proof_of_work(self,last_proof):
        proof = 0
        while self.valid_proof(last_proof,proof) is False:
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof,proof):
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hash.sha256(guess.encode()).hexdigest()
        return guess_hash[:4] == "0000"

    @staticmethod
    def hash(block):
        block_string = json.dumps(block,sort_keys=True).encode()
        return hash.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        return self.chain[-1]