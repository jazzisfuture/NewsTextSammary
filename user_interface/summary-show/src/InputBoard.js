import React from 'react';
import { Input,Button } from 'antd';

const { TextArea } = Input;

function InputBarod(){
    return(
    <div className="input-group">
        <TextArea rows={12}
            bordered='false'
            // autoSize='{ minRows: 6 }'
            style={{resize:'none'}}
        />
        <Button
          onClick={() => console.log("click")}
          style={{ marginTop: 16 }}
        >
          提交
        </Button>
    </div>
    )
}
export default InputBarod;