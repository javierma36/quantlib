/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2014 Mauricio Bedoya
 
 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/
 
 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.
 
 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
 */

/*! \file VGP.hpp
 \brief Variance Gamma stochastic process
 \source: The Variance Gamma Process and Option Pricing (Madan, Carr, Chang)
          Monte Carlo Methods in Financial Engineering (Glasserman)
          Monte Carlo Methods in Finance and Insurance (Korn, Korn, Kroisandt)
 */

#ifndef quantlib_variance_gamma_process_hpp
#define quantlib_variance_gamma_process_hpp

//#include <boost/random.hpp>
//#include <boost/random/gamma_distribution.hpp>
#include <ql/stochasticprocess.hpp>
#include <ql/processes/eulerdiscretization.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/termstructures/volatility/equityfx/blackvoltermstructure.hpp>
#include <ql/termstructures/volatility/equityfx/localvoltermstructure.hpp>
#include <ql/quote.hpp>

namespace QuantLib {
    
    //! Variance gamma process
    
    /*! This class describes the stochastic volatility process by 
     subordinator. Check source for more information.
     \f[
     X(0) = 0;
     dG(t) = gamrnd(dt/nu,nu)
     Z(t) = randn(0,1)
     omega = 1/ nu * [ln (1 - theta * nu - 0.5 * nu *sigma^2)]  
     X(t) = X(t-1) + theta * dG(t) + sigma * sqrt(dQ(t)) * Z(t)  (Return)
     S(t) = S(0) * e^(r - q + omega) + X(t)
     \f]
     \ingroup processes
     */
    class VarianceGammaProcess : public StochasticProcess1D {
    public:
        
        VarianceGammaProcess (const Handle<Quote>& s0,
                              const Handle<YieldTermStructure>& dividendYield,
                              const Handle<YieldTermStructure>& riskFreeRate,
                              const Handle<BlackVolTermStructure>& blackvolatility,
                              const Real& nu, const Real& theta,
                              const boost::shared_ptr<discretization>& d = boost::shared_ptr<discretization>(new EulerDiscretization));
        //@{
        /* Pure Virtual (StochasticProcess1D) */
        Real x0() const;
        Real drift(Time t, Real x) const;
        Real diffusion(Time t, Real x) const;
        /** Virtual (StochasticProcess1D) **/
        Real expectation(Time t0, Real x0, Time dt) const;
        Real evolve(Time t0, Real x0, Time dt, Real dg, Real dw) const; // Differs from virtual definition
        Real apply(Real x0, Real dx) const;
        /** Helper function **/
        Time time(const Date& ) const;
        void update();
        Real omega(Time t, Real x) const;
         /** Inspector **/
        Real nu() const { return nu_; }
        Real theta() const { return theta_; }
        const Handle<Quote>& s0() const;
        const Handle<YieldTermStructure>& dividendYield() const;
        const Handle<YieldTermStructure>& riskFreeRate() const;
        const Handle<BlackVolTermStructure>& blackVolatility() const;
        const Handle<LocalVolTermStructure>& localVolatility() const;
        
        //@}
    private:
        Handle<Quote> s0_;
        Handle<YieldTermStructure> dividendYield_, riskFreeRate_;
        Handle<BlackVolTermStructure> blackVolatility_;
        Real nu_, theta_;
        mutable bool updated_;
        mutable Real X0_;
        mutable RelinkableHandle<LocalVolTermStructure> localVolatility_;
    };
    
    
}

#endif


/**** .cpp file from here ****/

/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2014 Mauricio Bedoya
 
 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/
 
 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.
 
 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
 */

#include "variancegammaprocess.hpp"
#include <ql/termstructures/volatility/equityfx/localvolsurface.hpp>
#include <ql/termstructures/volatility/equityfx/localvolcurve.hpp>
#include <ql/termstructures/volatility/equityfx/localconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

namespace QuantLib {
    
    VarianceGammaProcess1::VarianceGammaProcess(const Handle<Quote>& s0,
                                                 const Handle<YieldTermStructure>& dividendYield,
                                                 const Handle<YieldTermStructure>& riskFreeRate,
                                                 const Handle<BlackVolTermStructure>& blackVolatility,
                                                 const Real& nu, const Real& theta,
                                                 const boost::shared_ptr<discretization>& disc)
    : StochasticProcess1D(disc),
    s0_(s0), dividendYield_(dividendYield), riskFreeRate_(riskFreeRate), blackVolatility_(blackVolatility), nu_(nu), theta_(theta),updated_(false), X0_(0) {
        registerWith(s0_);
        registerWith(dividendYield_);
        registerWith(riskFreeRate_);
        registerWith(blackVolatility_);
    }
    
    Real VarianceGammaProcess::x0() const
    {
        return s0_->value();
    }
    
    Real VarianceGammaProcess::drift(Time t, Real x) const
    {
        Time t1 = t + 0.0001;
        
        return riskFreeRate_->forwardRate(t, t1, Continuous, NoFrequency, true) - dividendYield_->forwardRate(t, t1, Continuous, NoFrequency, true) + omega(t,x);
    }
    
    Real VarianceGammaProcess::diffusion(Time t, Real x) const
    {
        return localVolatility()->localVol(t, x, true);
    }
    
    Real VarianceGammaProcess::expectation(Time t0, Real x0, Time dt) const {
        QL_FAIL("not implemented");
    }
    
    Real VarianceGammaProcess::evolve(Time t0, Real x0, Time dt, Real dg, Real dw) const {
        Real drift = discretization_->drift(*this,t0,x0,dt);
        X0_ += theta() * dg + stdDeviation(t0,x0,dt) * sqrt(dg) * dw;
        
        return apply(x0, drift + X0_);
    }
    
    Real VarianceGammaProcess::apply(Real x0, Real dx) const {
        return x0 * std::exp(dx);
    }
    
    Time VarianceGammaProcess::time(const Date& d) const
    {return riskFreeRate_->dayCounter().yearFraction(riskFreeRate_->referenceDate(), d);}
    
    void VarianceGammaProcess::update() {
        updated_ = false;
        StochasticProcess1D::update();
    }
    
    const Handle<Quote>& VarianceGammaProcess::s0() const {
        return s0_;
    }
    
    Real VarianceGammaProcess::omega(Time t, Real x) const {
        Real sigma = diffusion(t, x);
        return  1 / nu() * std::log(1 - theta() * nu() - sigma * sigma * nu() * 0.5);
    }
        
    const Handle<YieldTermStructure>& VarianceGammaProcess::dividendYield() const {
        return dividendYield_;
    }
    
    const Handle<YieldTermStructure>& VarianceGammaProcess::riskFreeRate() const {
        return riskFreeRate_;
    }
    
    const Handle<BlackVolTermStructure>& VarianceGammaProcess::blackVolatility() const{
        return blackVolatility_;
    }
    
    const Handle<LocalVolTermStructure>& VarianceGammaProcess::localVolatility() const{
        
        if(!updated_)
        {
            // Constant Volatility
            boost::shared_ptr<BlackConstantVol> constant = boost::dynamic_pointer_cast<BlackConstantVol>(*blackVolatility());
            if(constant)
            {
                localVolatility_.linkTo(boost::shared_ptr<LocalVolTermStructure>(new LocalConstantVol(constant->referenceDate(), constant->blackVol(0.0, s0_->value()), constant->dayCounter())));
                
                updated_ = true;
                return localVolatility_;
            }
            boost::shared_ptr<BlackVarianceCurve> volCurve = boost::dynamic_pointer_cast<BlackVarianceCurve>(*blackVolatility());
            if(volCurve)
            {
                localVolatility_.linkTo(boost::shared_ptr<LocalVolTermStructure>(new LocalVolCurve(Handle<BlackVarianceCurve>(volCurve))));
                
                updated_ = true;
                return localVolatility_;
            }
            // ok, so it's strike-dependent. Never mind.
            localVolatility_.linkTo(boost::shared_ptr<LocalVolTermStructure>(new LocalVolSurface(blackVolatility_,riskFreeRate_,dividendYield_, s0_->value())));
            
            updated_ = true;
            return localVolatility_;
        }
        else
        {
            return localVolatility_;
        }
    }
    
}



