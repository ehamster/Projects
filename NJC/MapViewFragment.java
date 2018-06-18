package com.parkingauthorityticketingsystem;

import android.app.AlertDialog;
import android.app.Fragment;
import android.app.FragmentTransaction;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.location.Location;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.MapView;
import com.google.android.gms.maps.MapsInitializer;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.model.CameraPosition;
import com.google.android.gms.maps.model.CircleOptions;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.LatLngBounds;
import com.google.android.gms.maps.model.Marker;
import com.google.android.gms.maps.model.MarkerOptions;
import com.google.android.gms.maps.model.PolylineOptions;
import com.parkingauthorityticketingsystem.database.ShiftlogDBHelper;
import com.parkingauthorityticketingsystem.entity.Shiftlog;
import com.parkingauthorityticketingsystem.utils.DialogGreenFragment;
import com.parkingauthorityticketingsystem.utils.DialogRedFragment;
import com.parkingauthorityticketingsystem.utils.DialogYellowFragment;
import com.parkingauthorityticketingsystem.utils.DirectionsJSONParser;
import com.parkingauthorityticketingsystem.utils.DistanceUtil;
import com.parkingauthorityticketingsystem.utils.DrawUtil;
import com.parkingauthorityticketingsystem.utils.Kmean;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;


/**
 * A simple {@link Fragment} subclass.
 */
public class MapViewFragment extends Fragment implements DialogRedFragment.DialogFragmentDBEdit,DialogGreenFragment.DialogFragmentDBEdit,
        DialogYellowFragment.DialogFragmentDBEdit {

    MapView mMapView;
    ArrayList<LatLng> markerPoints= new ArrayList<>(); // 2 points of the route
    private GoogleMap googleMap;
    private ShiftlogDBHelper helper;
    private TextView txtTime,txtDistance;
    private List<Shiftlog> shiftlogs = new ArrayList<>();
    private double x_coordinate;
    private double y_coordinate;
    private LatLng centroid;
    private double radius;
    private boolean showPath = false;
    LatLng start;
    @Override
    public View onCreateView(final LayoutInflater inflater, final ViewGroup container, Bundle savedInstanceState) {
        final View rootView = inflater.inflate(R.layout.fragment_map_view1, container, false);
        mMapView = (MapView) rootView.findViewById(R.id.mapView);
        mMapView.onCreate(savedInstanceState);
        mMapView.onResume(); // needed to get the map to display immediately

        try {
            MapsInitializer.initialize(getActivity().getApplicationContext());
        } catch (Exception e) {
            e.printStackTrace();
        }

        mMapView.getMapAsync(new OnMapReadyCallback() {
            @Override
            public void onMapReady(final GoogleMap mMap) {
                googleMap = mMap;

                // For showing a move to my location button
                if (ActivityCompat.checkSelfPermission(getContext(), android.Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED
                        && ActivityCompat.checkSelfPermission(getContext(), android.Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
                    // TODO: Consider calling
                    //    ActivityCompat#requestPermissions
                    // here to request the missing permissions, and then overriding
                    //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                    //                                          int[] grantResults)
                    // to handle the case where the user grants the permission. See the documentation
                    // for ActivityCompat#requestPermissions for more details.
                    return;
                }
                googleMap.setMyLocationEnabled(true);
                if (googleMap != null) {
                    googleMap.setOnMyLocationChangeListener(new GoogleMap.OnMyLocationChangeListener(){
                        public void onMyLocationChange(Location arg0) {
                            start = new LatLng(arg0.getLatitude(),arg0.getLongitude());
                        }
                    });
                }

                // For dropping a marker at a point on the Map
                x_coordinate = getArguments().getDouble("x");
                y_coordinate = getArguments().getDouble("y");
                LatLng sydney = new LatLng(x_coordinate, y_coordinate);
                //LatLng sydney = new LatLng(40, -74);
                // For zooming automatically to the location of the marker
                CameraPosition cameraPosition = new CameraPosition.Builder().target(sydney).zoom(16).build();
                googleMap.animateCamera(CameraUpdateFactory.newCameraPosition(cameraPosition));
                googleMap.setOnMarkerClickListener(new GoogleMap.OnMarkerClickListener() {
                    @Override
                    public boolean onMarkerClick(Marker marker) {
                        //get TimeLeft

                        Shiftlog sl = (Shiftlog) marker.getTag();
                        double timeLeftMilli = 0;
                        SimpleDateFormat formatter = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
                        Date curDate = new Date(System.currentTimeMillis());
                        try {
                            curDate = formatter.parse(sl.getFirstObserve());
                        } catch (ParseException e) {
                            e.printStackTrace();
                        }
                        timeLeftMilli = sl.getMaxTime()*60*1000 - (System.currentTimeMillis()- curDate.getTime());
                        timeLeftMilli = timeLeftMilli / 60 / 1000;
                        if(timeLeftMilli > 0) {
                            timeLeftMilli ++;
                        }
                        if(timeLeftMilli >= 6 ) {
                            showGreenDialog(sl,timeLeftMilli);
                        }
                        else if (timeLeftMilli < 6 && timeLeftMilli > 0){
                            showYellowDialog(sl,timeLeftMilli);
                        }
                        else{
                            showRedDialog(sl,timeLeftMilli);
                        }
                        return false;
                    }
                });
                helper = new ShiftlogDBHelper(getContext());
                shiftlogs = helper.queryAll(); // Read data from database
                // Display all marker
                DrawUtil.drawMarkers(shiftlogs,googleMap);
            }
        });
        rootView.findViewById(R.id.route_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (showPath == false) {
                    googleMap.clear();
                    DrawUtil.drawMarkers(shiftlogs,googleMap);
                   // start = new LatLng(x_coordinate, y_coordinate);
                    ///////////////////////////////////////////////////////////

                    ///////////////////////////////////////////////////////////
                    ArrayList<Shiftlog> sl = new ArrayList();
                    sl = helper.queryAll();


                /*for (int i = 0; i < sl.size(); i++) {
                    dataSet.add(new double[]{sl.get(i).getX_coordinate(),sl.get(i).getY_coordinate()});
                }*/
                    if (sl.size()>7){                                 //if more than 7 shiftlogs, do Kmean
                        int k_num = (int) Math.ceil((float)sl.size() / 7);
                        Kmean k=new Kmean(k_num);
                        ArrayList<Shiftlog> dataSet=new ArrayList();
                        k.setDataSet(sl);
                        k.execute();
                        ArrayList<double[]> center = k.getCenter();
                   /* double distance = DistanceUtil.getDistance(start,center.get(0)[0],center.get(0)[1]);

                    for(int i=1;i<center.size();i++)
                    {

                        if (DistanceUtil.getDistance(start,center.get(i)[0],center.get(i)[1]) < distance){
                            index = i;
                        }
                    }
                    */
                        ArrayList<ArrayList<Shiftlog>> cluster = k.getCluster();
                        int sum_weight = 0;
                        ArrayList<Integer> cluster_weight = new ArrayList<>();

                        //Calculate weight of clusters
                        for (int i = 0; i < cluster.size(); i ++) {
                            for (int j = 0; j < cluster.get(i).size(); j++) {
                                long timeLeft = 0;
                                sum_weight = 0;
                                SimpleDateFormat formatter = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
                                Date curDate = new Date(System.currentTimeMillis());
                                try {
                                    curDate = formatter.parse(cluster.get(i).get(j).getFirstObserve());
                                } catch (ParseException e) {
                                    e.printStackTrace();
                                }
                                timeLeft = cluster.get(i).get(j).getMaxTime()*60*1000 - (System.currentTimeMillis()- curDate.getTime());
                                timeLeft = timeLeft / 1000;
                                if (timeLeft >= 240 && timeLeft <= 600) {
                                    sum_weight += 3;
                                }else if (timeLeft >= -120 && timeLeft < 240) {
                                    sum_weight += 5;
                                }else if (timeLeft >= -480 && timeLeft < -120) {
                                    sum_weight += 4;
                                }else if (timeLeft >= -840 && timeLeft < -480) {
                                    sum_weight += 2;
                                }else if (timeLeft >= -1200 && timeLeft < -840) {
                                    sum_weight += 1;
                                }else {

                                }
                            }
                            cluster_weight.add(sum_weight);

                        }
                        Integer max = cluster_weight.get(0);
                        int index_weight = 0;
                        for (int i = 0; i < cluster_weight.size(); i++) {
                            if (max < cluster_weight.get(i)) {
                                max = cluster_weight.get(i);
                                index_weight = i;
                            }
                        }
                        ArrayList<Integer> max_weight_index = new ArrayList<>();
                        for (int i = 0; i < cluster_weight.size(); i ++) {  //Get max weight clusters index
                            if (cluster_weight.get(i) == max) {
                                max_weight_index.add(i);
                            }
                        }
                        Log.i("index","max" + max);
                        Log.i("index1","index_weight" + index_weight);
                        double minDistance = DistanceUtil.getDistance(start.latitude,start.longitude,
                                center.get(max_weight_index.get(0))[0],
                                center.get(max_weight_index.get(0))[1]);
                        for (int i = 0; i < max_weight_index.size(); i ++) {
                            if (DistanceUtil.getDistance(start.latitude,start.longitude,
                                    center.get(max_weight_index.get(i))[0],
                                    center.get(max_weight_index.get(i))[1]) < minDistance){
                                minDistance = DistanceUtil.getDistance(start.latitude,start.longitude,
                                        center.get(max_weight_index.get(i))[0],
                                        center.get(max_weight_index.get(i))[1]);
                                index_weight = i;
                            }
                        }
                        Log.i("index2","MinDistance" + minDistance);
                        centroid = new LatLng(center.get(index_weight)[0],center.get(index_weight)[1]);
                        Log.i("cluster",sl.get(0).getX_coordinate() + " " +sl.get(0).getY_coordinate() );
                        radius = DistanceUtil.getDistance(cluster.get(index_weight).get(0).getX_coordinate(),
                                cluster.get(index_weight).get(0).getY_coordinate(),
                                center.get(index_weight)[0],
                                center.get(index_weight)[1]);
                        for (int i = 0; i < cluster.get(index_weight).size(); i++) {
                            double distance = DistanceUtil.getDistance(cluster.get(index_weight).get(i).getX_coordinate(),
                                    cluster.get(index_weight).get(i).getY_coordinate(),
                                    center.get(index_weight)[0],
                                    center.get(index_weight)[1]);
                            radius = Math.max(radius,distance);
                        }
                        Log.i("cluster",cluster.size()+"");
                        LatLng end = new LatLng(centroid.latitude, centroid.longitude); // route from local to centroid of cluster
                        // Getting URL to the Google Directions API
                        //end = new LatLng(40.748,-74.151);
                        String url = getDirectionsUrl(start, end);

                        DownloadTask downloadTask = new DownloadTask();

                        // Start downloading json data from Google Directions API
                        downloadTask.execute(url);
                        showPath = true;
                    }
                    else if (sl.size() == 0){
                        Toast.makeText(getContext(),"No Log Entry",Toast.LENGTH_SHORT).show();
                    }
                    else{
                        sl = helper.queryAll();
                        int index = 0;
                        radius = 0;
                        double distance =  DistanceUtil.getDistance(start.latitude,start.longitude,sl.get(0).getX_coordinate(),sl.get(0).getY_coordinate());
                        for (int i = 0; i < sl.size(); i++) {
                            if (DistanceUtil.getDistance(start.latitude,start.longitude,sl.get(i).getX_coordinate(),sl.get(i).getY_coordinate()) < distance){
                                distance = DistanceUtil.getDistance(start.latitude,start.longitude,sl.get(i).getX_coordinate(),sl.get(i).getY_coordinate());
                                index = i;
                            }
                        }
                        LatLng end = new LatLng(sl.get(index).getX_coordinate(), sl.get(index).getY_coordinate()); // route from local to centroid of cluster
                        // Getting URL to the Google Directions API
                        String url = getDirectionsUrl(start, end);

                        DownloadTask downloadTask = new DownloadTask();

                        // Start downloading json data from Google Directions API
                        downloadTask.execute(url);
                       // Toast.makeText(getContext(),"less",Toast.LENGTH_SHORT).show();
                        showPath = true;
                        }
                    }
                else {   //Path already exist
                    new AlertDialog.Builder(getContext()).setTitle("Warning").setMessage("Do you want to clear the Route ?")
                            .setNegativeButton("Confirm", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    googleMap.clear();
                                    DrawUtil.drawMarkers(shiftlogs,googleMap);
                                    showPath = false;
                                }
                            }).setPositiveButton("Cancel", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {

                        }
                    }).create().show();
                }




            }
        });
        rootView.findViewById(R.id.viewshiftlog_btn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
              System.out.println("Change to List View");
//                SL_ListViewFragment listViewFragment = new SL_ListViewFragment();
//                getActivity().getFragmentManager().beginTransaction().replace(R.id.fragmentContainer,listViewFragment)
//                        .addToBackStack(null).commit();
                getActivity().getFragmentManager().beginTransaction().replace(R.id.fragmentContainer,new SL_ListViewFragment(),"ListViewFrag")
                        .addToBackStack(null).commit();
            }

        });

        return rootView;
    }
    /////////////////////////////////////////////

    private class DownloadTask extends AsyncTask<String, Void, String> {

        @Override
        protected String doInBackground(String... url) {

            String data = "";

            try {
                data = downloadUrl(url[0]);
            } catch (Exception e) {
                Log.d("Background Task", e.toString());
            }
            return data;
        }

        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);

            ParserTask parserTask = new ParserTask();


            parserTask.execute(result);

        }
    }
    private class ParserTask extends AsyncTask<String, Integer, List<List<HashMap<String, String>>>> {

        // Parsing the data in non-ui thread
        @Override
        protected List<List<HashMap<String, String>>> doInBackground(String... jsonData) {

            JSONObject jObject;
            List<List<HashMap<String, String>>> routes = null;

            try {
                jObject = new JSONObject(jsonData[0]);
                DirectionsJSONParser parser = new DirectionsJSONParser();

                routes = parser.parse(jObject);
            } catch (Exception e) {
                e.printStackTrace();
            }
            return routes;
        }

        @Override
        protected void onPostExecute(List<List<HashMap<String, String>>> result) {
            ArrayList points = null;
            PolylineOptions lineOptions = new PolylineOptions();

            lineOptions.width(18);
            lineOptions.color(Color.argb(98,41,69,225));
            MarkerOptions markerOptions = new MarkerOptions();

            points = new ArrayList();
            try {
                List<HashMap<String, String>> path = result.get(0);
                List<HashMap<String, String>> distance = result.get(1);
                List<HashMap<String, String>> duration = result.get(2);

                // To draw path between two location
                drawPath(path);
                for (int j = 0; j < path.size(); j++) {
                    HashMap<String, String> point = path.get(j);

                    double lat = Double.parseDouble(point.get("lat"));
                    double lng = Double.parseDouble(point.get("lng"));
                    LatLng position = new LatLng(lat, lng);

                    points.add(position);
                }

                lineOptions.addAll(points);
                lineOptions.geodesic(true);

                // To count distance between two location
                double totalDistance = 0;
                for (int j = 0; j < distance.size(); j++) {
                    HashMap<String, String> point = distance.get(j);

                    int dist = Integer.parseInt(point.get("distance"));
                    totalDistance = totalDistance + (double)dist;
                }
                totalDistance = totalDistance / 1000.0;
               /* String totalDist = "Total distance between location is:" +
                        String.valueOf(totalDistance + "Km");*/
                ((TextView) getActivity().findViewById(R.id.txtDistance)).setText(totalDistance + "km");//KM
                Log.i("Distance",txtDistance + "");

                // To count duration between two location
                int totalDuration = 0;
                for (int k = 0; k < duration.size(); k++) {
                    HashMap<String, String> point = duration.get(k);

                    int dist = Integer.parseInt(point.get("duration"));
                    totalDuration = totalDuration + dist;
                }
                convertSecondToHour(totalDuration);
            } catch (IndexOutOfBoundsException e) {
                e.printStackTrace();
            }

            // for covering the path drawn in display
            boolean hasPoints = false;
            Double maxLat = null, minLat = null, minLon = null, maxLon = null;

            if (lineOptions != null && lineOptions.getPoints() != null) {
                List<LatLng> pts = lineOptions.getPoints();
                for (LatLng coordinate : pts) {
                    // Find out the maximum and minimum latitudes & longitudes
                    // Latitude
                    maxLat = maxLat != null ? Math.max(coordinate.latitude, maxLat) : coordinate.latitude;
                    minLat = minLat != null ? Math.min(coordinate.latitude, minLat) : coordinate.latitude;

                    // Longitude
                    maxLon = maxLon != null ? Math.max(coordinate.longitude, maxLon) : coordinate.longitude;
                    minLon = minLon != null ? Math.min(coordinate.longitude, minLon) : coordinate.longitude;

                    hasPoints = true;
                }
            }

            if (hasPoints) {
                LatLngBounds.Builder builder = new LatLngBounds.Builder();
                builder.include(new LatLng(maxLat, maxLon));
                builder.include(new LatLng(minLat, minLon));
                googleMap.moveCamera(CameraUpdateFactory.newLatLngBounds(builder.build(), 48));
                googleMap.addPolyline(lineOptions);
                if (radius != 0) {
                    googleMap.addCircle(new CircleOptions().center(new LatLng(centroid.latitude, centroid.longitude))
                            .radius(radius * 111111 + 10).strokeColor(Color.argb(98, 41, 69, 225)).fillColor(0x220000FF));
                }
                else {
                }
                Log.i("radius",radius + "");
            }
        }
    }

    private void drawPath(List<HashMap<String, String>> path) {
    }
    private void convertSecondToHour(int totalDuration) {
        int hours = totalDuration / 3600;
        int minutes = (totalDuration % 3600) / 60;
        int seconds = (totalDuration % 3600) % 60;

       /* String totalTime = "Total duration between location is:" + String.valueOf(hours) + ":" +
                String.valueOf(minutes) + ":" + String.valueOf(seconds);*/
        ((TextView)getActivity().findViewById(R.id.txtTime)).setText(minutes + "min");

        Log.i("needTime",txtTime + "");
    }
    private String getDirectionsUrl(LatLng origin, LatLng dest) {

        // Origin of route
        String str_origin = "origin=" + origin.latitude + "," + origin.longitude;

        // Destination of route
        String str_dest = "destination=" + dest.latitude + "," + dest.longitude;

        // Sensor enabled
        String sensor = "sensor=false";
        String mode = "mode=walking";

        // Building the parameters to the web service
        String parameters = str_origin + "&" + str_dest + "&" + sensor + "&" + mode;

        // Output format
        String output = "json";

        // Building the url to the web service
        String url = "https://maps.googleapis.com/maps/api/directions/" + output + "?" + parameters;


        return url;
    }
    private String downloadUrl(String strUrl) throws IOException {
        String data = "";
        InputStream iStream = null;
        HttpURLConnection urlConnection = null;
        try {
            URL url = new URL(strUrl);

            urlConnection = (HttpURLConnection) url.openConnection();

            urlConnection.connect();

            iStream = urlConnection.getInputStream();

            BufferedReader br = new BufferedReader(new InputStreamReader(iStream));

            StringBuffer sb = new StringBuffer();

            String line = "";
            while ((line = br.readLine()) != null) {
                sb.append(line);
            }

            data = sb.toString();

            br.close();

        } catch (Exception e) {
            Log.d("Exception", e.toString());
        } finally {
            iStream.close();
            urlConnection.disconnect();
        }
        return data;

    }
    private void showRedDialog(Shiftlog sl, double timeLeftMilli) {
        FragmentTransaction mFragTransaction = getFragmentManager().beginTransaction();
        Fragment fragment =  getFragmentManager().findFragmentByTag("dialogFragmentRed");
        if(fragment!=null){
            //Avoid display dialog twice
            mFragTransaction.remove(fragment);
        }
        DialogRedFragment dialogFragment =new DialogRedFragment().newInstance(sl,timeLeftMilli);
        dialogFragment.setDialogFragmentDBEdit(MapViewFragment.this);
        dialogFragment.show(mFragTransaction,"dialogFragmentRed");


    }

    private void showGreenDialog(Shiftlog sl,double timeLeftMilli) {
        FragmentTransaction mFragTransaction = getFragmentManager().beginTransaction();
        Fragment fragment =  getFragmentManager().findFragmentByTag("dialogFragment");
        if(fragment!=null){
            //Avoid display dialog twice
            mFragTransaction.remove(fragment);
        }
        DialogGreenFragment dialogFragment =DialogGreenFragment.newInstance(sl,timeLeftMilli);
        dialogFragment.setDialogFragmentDBEdit(MapViewFragment.this);
        dialogFragment.show(mFragTransaction,"dialogFragment");
    }
    private void showYellowDialog(Shiftlog sl,double timeLeftMilli) {
        FragmentTransaction mFragTransaction = getFragmentManager().beginTransaction();
        Fragment fragment =  getFragmentManager().findFragmentByTag("dialogFragmentYellow");
        if(fragment!=null){
            //Avoid display dialog twice
            mFragTransaction.remove(fragment);
        }
        DialogYellowFragment dialogFragment = DialogYellowFragment.newInstance(sl,timeLeftMilli);
        dialogFragment.setDialogFragmentDBEdit(MapViewFragment.this);
        dialogFragment.show(mFragTransaction,"dialogFragmentYellow");
    }


    @Override
    public void onResume() {
        super.onResume();
        mMapView.onResume();
    }

    @Override
    public void onPause() {
        super.onPause();
        mMapView.onPause();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        mMapView.onDestroy();
    }

    @Override
    public void onLowMemory() {
        super.onLowMemory();
        mMapView.onLowMemory();
    }

    @Override
    public void remove(final int id) {  //rewrite remove() in interface, in DialogRedFragment use this method
        helper = new ShiftlogDBHelper(getContext());
        new AlertDialog.Builder(getContext()).setTitle("Warning").setMessage("Are you sure to remove ?")
                .setNegativeButton("Confirm", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        helper.deleteRecord(id);
                        googleMap.clear();
                        for(Shiftlog sl : shiftlogs) {
                            if (sl.getId() == id) {
                                shiftlogs.remove(sl);
                                break;
                            }
                        }
                        DrawUtil.drawMarkers(shiftlogs,googleMap);
                    }
                }).setPositiveButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {

            }
        }).create().show();


    }

    public void ticket(Shiftlog sl) {
      /*  helper.deleteRecord(id);
        googleMap.clear();
        for(Shiftlog sl : shiftlogs) {
            if (sl.getId() == id) {
                shiftlogs.remove(sl);
                break;
            }
        }
        DrawUtil.drawMarkers(shiftlogs,googleMap);*/
    }


}
