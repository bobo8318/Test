from uuid import uuid4
from flask import Flask, jsonify, request
from BlockChain import BlockChain
import pymysql

app = Flask(__name__)
#为节点创建一个随机名称
node_identifier = str(uuid4()).replace('-','')
blockchain = BlockChain()

@app.route('/')
def hello_world():
    return 'Hello My BlockChain!'




@app.route('/mine',methods=['GET'])
def mine():
    last_block = blockchain.last_block
    #上一个块的 工作量证明 值
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)

    blockchain.new_transaction(
        sender = "0",
        recipient = node_identifier,
        amount = 1,
    )

    prev_hash = blockchain.hash(last_block)
    block = blockchain.new_block(proof,prev_hash)
    response = {
        'message':"new block forged",
        'index':block['index'],
        'transactions':block['transactions'],
        'proof':block['proof'],
        'prev_hash':block['prev_hash'],
    }
    return jsonify(response),200


# 给接口发送交易数据
@app.route('/transactions/new',methods=['POST'])
def new_transaction():
    values = request.get_json()
    required = ['sender','recipient','amount']
    if not all(k in values for k in required):
        return 'Missing values',400
    # Create a new Transaction
    index = blockchain.new_transaction(values['sender'], values['recipient'], values['amount'])

    response = {'message': f'Transaction will be added to Block {index}'}
    return jsonify(response), 201

 # 返回整个区块链
@app.route('/chain',methods=['GET'])
def full_chain():
    response = {
        'chain':blockchain.chain,
        'length':len(blockchain.chain),
    }
    print('full chain')
    return jsonify(response),200

@app.route('/register',methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')

    if nodes is None:
        return "Error:Please supply a valid list of nodes",400

    for node in nodes:
        blockchain.register_node(node)

    response = {
        'message':'New nodes have been added',
        'total_nodes':list(blockchain.nodes),
    }
    return jsonify(response),201

@app.route('/nodes/resolve',methods={'GET'})
def consensus():
    replaced = blockchain.resolve_conflicts()
    if replaced:
        response = {
            'message':'Our chain was replaced',
            'chain':blockchain.chain
        }
    else:
        response = {
            'message':'Our chain is authoritative',
            'chain':blockchain.chain
        }
    return jsonify(response),200


if __name__ == '__main__':
    app.run()