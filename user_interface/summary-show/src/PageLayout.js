import React from 'react';
import ResBoard from './ResBoard.js';
import { Layout, Menu} from 'antd';
import './CSS/PageLayout.css'
const { Header, Content, Footer } = Layout;


function PageLayout(){
    return(
        <Layout>
    <Header style={{ position: 'fixed', zIndex: 1, width: '100%' }}>
      <div className="logo" />
      <Menu theme="dark" mode="horizontal">
        <Menu.Item key="1" style={{fontSize:30}}>基于Transformer的文本摘要</Menu.Item>
      </Menu>
    </Header>
    <Content className="site-layout" style={{ padding: '0 50px', marginTop: 64 }}>
      <div className="site-layout-background" style={{ padding: 24, minHeight: 800 }}>
        <ResBoard/>
      </div>
    </Content>
    <Footer style={{ textAlign: 'center' }}>Jiang Yulong ©2021</Footer>
  </Layout>
    );
};
export default PageLayout;