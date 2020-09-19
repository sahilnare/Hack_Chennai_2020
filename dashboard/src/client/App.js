import React, { Component, Suspense, lazy } from 'react';
import './app.css';
import { BrowserRouter, Switch, Route, Redirect } from 'react-router-dom';
const Rakshak = lazy(() => import('./pages/cyberally/CyberAlly' /* webpackChunkName: "cyberally" */));
import { connect } from 'react-redux';



class App extends Component {

  constructor(props) {
    super(props);
  }

  componentDidMount() {
  }


  render() {

    return (
      <BrowserRouter>

        <main style={{minHeight: '100vh'}}>
        {
          isLoading ? (
            <div className='loader'>Loading...</div>
          ) : (
            <Suspense fallback={<div className='loader'>Loading...</div>}>
              <Switch>
                <Route
                  path='/rakshak'
                  exact={true}
                  render={(props) => <Rakshak {...props} />}
                />
              </Switch>
            </Suspense>
          )
        }
        </main>
      </BrowserRouter>
    );
  }
}


export default App;
