import React,{Component} from 'react';
import { Input,Button, Card} from 'antd';
import './CSS/ResBoard.css'
import {get,post} from './api/http.js'
import api from './api/api.js'
import './index.css';
const { TextArea } = Input;
Component.prototype.get = get;
Component.prototype.post = post;
Component.prototype.api = api;



class Summary extends Component {
    constructor(props){
        super(props);
        this.state={
            text:"",
            result:"",
        }
    }
    submitText = () => {
        console.log(this.state)
        this.post(`${this.api.summary}`,{'text':this.state.text}).then(res=>{
            this.setState({result:res.res})
        })
    }
    // 更改
    changeText=(event)=>{
        this.setState({text:event.target.value})
    }
    changeRes=(event)=>{
        this.setState({result:event.target.value})
    }

    render(){
        return(
            <div>
            <div className="input-group">
                <TextArea placeholder='请输入需要摘要的文章'
                          bordered='false'
                          autoSize={{ minRows: 8, maxRows: 14 }}
                          style={{resize:'none'}}
                          onChange={(e)=>this.changeText(e)}
                />
                <Button
                    onClick={() => this.submitText()}
                    style={{ marginTop: 16, marginBottom:16}}
                >
                    提交
                </Button>
            </div>
            <div className="result">
            <Card   title="结果"
                    headStyle={{fontSize:22}}
                    hoverable='true'        
            >
                <p>{this.state.result.trim().replace('(图 )','')}</p>
            </Card>
            </div>
            </div>
            
        )
    }
} 

export default Summary;