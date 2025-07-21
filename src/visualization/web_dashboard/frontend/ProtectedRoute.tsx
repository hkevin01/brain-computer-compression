import React from 'react';
import { Route, Redirect } from 'react-router-dom';

// ProtectedRoute component for authenticated dashboard access
const ProtectedRoute: React.FC<{ sessionId: string; path: string; component: React.FC }> = ({ sessionId, path, component: Component }) => {
  return (
    <Route
      path={path}
      render={props =>
        sessionId ? <Component {...props} /> : <Redirect to="/login" />
      }
    />
  );
};

export default ProtectedRoute;
